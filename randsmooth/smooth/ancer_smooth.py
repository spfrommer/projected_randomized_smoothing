import torch
import torch.nn.functional as F

import math
import numpy as np
from scipy import stats
from statsmodels.stats import proportion

from ddsmoothing.certificate import L2Certificate
from ddsmoothing.optimization import optimize_isotropic_dds
from ancer.optimization import optimize_ancer

from randsmooth.utils import torch_utils, math_utils
from randsmooth.smooth.smooth import Smooth


class AncerSmooth(Smooth):
    def __init__(self, base, name, sigma=0.25, n0=100, n=1000, alpha=0.001,
                 ancer_n=100, ancer_iterations=100, ancer_lr=0.04,
                 ancer_kappa=2.0, isotropic_iterations=900):
        super().__init__(base, name)

        # Ancer expects probability simplex, output of base is logits
        self.base_softmax = torch_utils.SoftmaxWrapper(self.base)

        self.sigma = sigma
        self.n0 = n0
        self.n = n
        self.alpha = alpha

        self.ancer_n = ancer_n
        self.ancer_iterations = ancer_iterations
        self.ancer_lr = ancer_lr
        self.ancer_kappa = ancer_kappa

        self.isotropic_iterations = isotropic_iterations

    def predict(self, x):
        self.base.eval()
        thetas = self.optimize_thetas(x)
        counts = self.sample_under_noise(x, self.n, thetas)
        (na, nb), (ca, cb) = torch.topk(counts, 2)

        if stats.binomtest(na, na + nb, 0.5).pvalue <= self.alpha:
            return ca

        return torch.tensor(-1).type_as(x)  # Abstain

    def certify(self, x):
        self.base.eval()

        thetas = self.optimize_thetas(x)

        counts0 = self.sample_under_noise(x, self.n0, thetas)
        ca = torch.argmax(counts0)
        counts = self.sample_under_noise(x, self.n, thetas)

        pa = proportion.proportion_confint(torch_utils.numpy(counts[ca]), self.n,
                                           alpha=2 * self.alpha, method='beta')[0]

        if pa > 0.5:
            rs = thetas.reshape(-1) * stats.norm.ppf(pa)
            in_n = np.prod(list(x.shape[1:]))
            vol_log = math_utils.nellipse_vol_log(in_n, rs)

            boundary_n = (x.abs() > 0.49999).sum()
            vol_log_clipped = vol_log - boundary_n.item() * math.log(2)

            return ca, vol_log, {'radius': rs.mean().item(), 'clip_vol_log': vol_log_clipped}

        return torch.tensor(-1).type_as(x), -math.inf, {'radius': 0, 'clip_vol_log': -math.inf}

    def sample_under_noise(self, x, n, thetas, count=True):
        assert x.shape[0] == 1
        assert x.shape == thetas.shape

        def get_preds(n_sub):
            noise = torch.randn([n_sub] + list(x.shape[1:])).to(torch_utils.device())
            samples = x + noise * thetas
            pred = self.base(samples)
            if count:
                pred = F.one_hot(pred.argmax(1), num_classes=pred.shape[1])
            return torch.sum(pred, dim=0)

        max_n = 2000
        full_batches, remainder = n // max_n, n % max_n

        batches = [get_preds(max_n) for _ in range(full_batches)]
        if remainder > 0:
            batches = batches + [get_preds(remainder)]
        pred = sum(batches)
        return pred

    def optimize_thetas(self, x):
        with torch.enable_grad():
            thetas = torch.tensor(self.sigma).to(torch_utils.device())
            cert = L2Certificate(batch_size=x.shape[0], device=torch_utils.device())
            iso_thetas = optimize_isotropic_dds(
                self.base_softmax, x, cert,
                self.ancer_lr, thetas, self.isotropic_iterations, self.ancer_n,
                torch_utils.device()
            )
            ancer_thetas = torch.ones_like(x).to(torch_utils.device()) \
                * iso_thetas.reshape(-1, 1, 1, 1)
            ancer_thetas = optimize_ancer(
                self.base_softmax, x, cert,
                self.ancer_lr, ancer_thetas,
                self.ancer_iterations, self.ancer_n, self.ancer_kappa,
                torch_utils.device()
            )

        return ancer_thetas
