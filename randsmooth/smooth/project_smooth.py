import torch

import math
import numpy as np
import mosek.fusion as mf
from scipy import stats

import rs4a.src.smooth as rs4a_smooth
import rs4a.src.noises as rs4a_noises

from randsmooth.utils import torch_utils, math_utils
from randsmooth.smooth.smooth import Smooth


class ProjectSmooth(Smooth):
    def __init__(self, base, name, sigma=0.25, n0=100, n=1000, alpha=0.001):
        super().__init__(base, name)

        project_U = self.base.project.project_U

        self.noise = rs4a_noises.Gaussian(project_U.shape[0],
                                          sigma=sigma, device=torch_utils.device())
        self.n0 = n0
        self.n = n
        self.alpha = alpha

        # Pre-computed problem
        project_null = torch_utils.complete_basis(project_U)
        project_null = torch_utils.numpy(project_null).astype(np.double)
        self.problem = self.mosek_construct_min_infty_norm(project_null)

    def compose_out(self, shape_init):
        return lambda x: self.base.net(self.unproject(x, shape_init))

    def predict(self, x):
        shape_init = x.shape

        self.base.eval()
        x = self.project(x)

        netout = self.compose_out(shape_init)
        counts = rs4a_smooth.smooth_predict_hard(netout, x, self.noise, self.n, raw_count=True)
        (na, nb), (ca, cb) = torch.topk(counts.int(), 2)

        if stats.binomtest(na, na + nb, 0.5).pvalue <= self.alpha:
            return ca

        return torch.tensor(-1).type_as(x)

    def certify(self, x):
        assert x.shape[0] == 1
        shape_init = x.shape
        x_init = x.reshape(1, -1)
        self.base.eval()
        x = self.project(x)

        netout = self.compose_out(shape_init)
        preds = rs4a_smooth.smooth_predict_hard(netout, x, self.noise, self.n0)
        top_cats = preds.probs.argmax(dim=1)
        prob_lb = rs4a_smooth.certify_prob_lb(
            netout, x, top_cats, 2 * self.alpha, self.noise, self.n)

        if prob_lb > 0.5:
            r = self.noise.certify_l2(prob_lb)

            project_n, in_n = self.base.project.project_U.shape

            x_init = torch_utils.numpy(x_init.squeeze(0)).astype(np.double)

            t = self.mosek_solve_min_infty_norm(x_init)

            r_star = min(r, project_n * (1 - 2 * t) / (2 * in_n))

            if r_star <= 0.00001 or r_star >= (1/2 - t) - 0.00001:
                # Can happen if can't get better point in plane than infty norm of 0.5
                return torch.tensor(-1).type_as(x), -math.inf, {}

            ball_vol_log = math_utils.nball_vol_log(project_n, r_star)

            extrude_vol = (in_n - project_n) * math.log(1 - 2 * r_star - 2 * t)

            cyl_vol_log = ball_vol_log + extrude_vol

            return top_cats, cyl_vol_log, {'radius': r.item()}

        return torch.tensor(-1).type_as(x), -math.inf, {}

    def mosek_construct_min_infty_norm(self, project_null):
        plane_k, in_n = project_null.shape

        M = mf.Model()
        x_param = M.parameter('x', in_n)

        alpha, t = M.variable(plane_k), M.variable()

        x_star = mf.Expr.add(x_param, mf.Expr.mul(alpha, project_null))

        t_reshape = mf.Expr.mul(t, np.ones(in_n))
        M.constraint(mf.Expr.add(mf.Expr.neg(x_star), t_reshape), mf.Domain.greaterThan(0.0))
        M.constraint(mf.Expr.add(x_star, t_reshape), mf.Domain.greaterThan(0.0))

        M.objective(mf.ObjectiveSense.Minimize, t)

        M.setSolverParam('presolveUse', 'off')
        M.setSolverParam('intpntBasis', 'never')

        return M

    def mosek_solve_min_infty_norm(self, x):
        x_param = self.problem.getParameter('x')
        x_param.setValue(x)
        self.problem.solve()

        if self.problem.getProblemStatus() != mf.ProblemStatus.PrimalAndDualFeasible:
            raise RuntimeError('Infeasible optimization!')

        return self.problem.primalObjValue()

    def project(self, x):
        batch_n = x.shape[0]
        x = x.reshape(batch_n, -1)
        return x @ self.base.project.project_U.T

    def unproject(self, x, shape_init):
        batch_n = x.shape[0]
        x = x @ self.base.project.project_U
        return x.reshape(batch_n, *shape_init[1:])
