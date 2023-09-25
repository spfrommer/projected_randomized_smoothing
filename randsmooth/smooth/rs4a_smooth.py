import torch

import math
import numpy as np
from scipy import stats

import rs4a.src.smooth as rs4a_smooth

from randsmooth.utils import math_utils
from randsmooth.smooth.smooth import Smooth


class RS4ASmooth(Smooth):
    def __init__(self, base, name, noise, ball_norm, n0=100, n=1000, alpha=0.001):
        super().__init__(base, name)

        self.noise = noise
        self.ball_norm = ball_norm
        self.n0 = n0
        self.n = n
        self.alpha = alpha

    def predict(self, x):
        self.base.eval()
        counts = rs4a_smooth.smooth_predict_hard(self.base, x, self.noise, self.n, raw_count=True)
        (na, nb), (ca, cb) = torch.topk(counts.int(), 2)

        if stats.binomtest(na, na + nb, 0.5).pvalue <= self.alpha:
            return ca

        return torch.tensor(-1).type_as(x)

    def certify(self, x):
        self.base.eval()
        preds = rs4a_smooth.smooth_predict_hard(self.base, x, self.noise, self.n0)
        top_cats = preds.probs.argmax(dim=1)
        prob_lb = rs4a_smooth.certify_prob_lb(
            self.base, x, top_cats, 2 * self.alpha, self.noise, self.n)

        if prob_lb > 0.5:
            in_n = np.prod(list(x.shape[1:]))
            if self.ball_norm == 'l1':
                r = self.noise.certify_l1(prob_lb)
                vol_log = math_utils.nball_l1_vol_log(in_n, r)
            elif self.ball_norm == 'l2':
                r = self.noise.certify_l2(prob_lb)
                vol_log = math_utils.nball_vol_log(in_n, r)
            elif self.ball_norm == 'linf':
                r = self.noise.certify_linf(prob_lb)
                vol_log = math_utils.nball_linf_vol_log(in_n, r)

            boundary_n = (x.abs() > 0.49999).sum()
            vol_log_clipped = vol_log - boundary_n.item() * math.log(2)

            return top_cats, vol_log, {'radius': r.item(), 'clip_vol_log': vol_log_clipped}

        return torch.tensor(-1).type_as(x), -math.inf, {'radius': 0, 'clip_vol_log': -math.inf}
