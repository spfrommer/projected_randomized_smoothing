import torch
import torch.nn as nn

from randsmooth.utils import torch_utils

import mosek
from mosek.fusion import *

from torchattacks.attack import Attack


class SubspacePGD(Attack):
    # Adapted from https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/torchattacks/attacks/pgd.py

    def __init__(self, model, subspace,
                 eps=8/255, alpha=2/255, steps=40, random_start=True, grad_sign=True):
        super().__init__('SubspacePGD', model)
        self.subspace = subspace
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.grad_sign = grad_sign
        self._supported_mode = ['default']

        self.problem = self.construct_project()

    def forward(self, images, labels):
        assert images.shape[0] == 1

        subspace_n, in_n = self.subspace.shape

        images = images - 0.5 # Make consistent with rest of interface

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        loss = nn.CrossEntropyLoss()

        if self.random_start:
            # Starting at a uniformly random point (eps is not meaningful after pojection)
            w = torch.zeros(subspace_n).uniform_(-self.eps, self.eps).to(torch_utils.device())
            w = self.project_w_star(images, w)
            if w is None:
                return images

        for _ in range(self.steps):
            w.requires_grad = True
            adv_images = images + (self.subspace.T @ w).reshape(1, *images.shape[1:])
            outputs = self.model(adv_images)

            # Calculate loss
            cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, w, retain_graph=False, create_graph=False)[0]

            if self.grad_sign:
                w_star = w.detach() + self.alpha * grad.sign()
            else:
                w_star = w.detach() + self.alpha * grad

            w = self.project_w_star(images, w_star)
            if w is None:
                return images

        final_delta = (self.subspace.T @ w).reshape(1, *images.shape[1:])
        # assert final_delta.abs().max() <= self.eps + 0.00001
        # assert (images + final_delta).abs().max() <= 1/2 + 0.00001

        # print(f'Start loss: {loss(self.model(images), labels)}')
        # print(f'Final loss: {loss(self.model(images + final_delta), labels)}')

        return images + final_delta + 0.5

    def project_w_star(self, image, w_star):
        self.problem.getParameter('w_star').setValue(torch_utils.numpy(w_star.double()))
        self.problem.getParameter('x').setValue(torch_utils.numpy(image.reshape(-1).double()))

        self.problem.solve()

        if self.problem.getProblemStatus() != ProblemStatus.PrimalAndDualFeasible:
            print('Infeasible subspace projection optimization')
            return None

        w = torch.tensor(self.problem.getVariable('w').level(), device=torch_utils.device())
        return w.float()

    def construct_project(self):
        U = torch_utils.numpy(self.subspace.double())
        subspace_n, in_n = self.subspace.shape

        M = Model()
        w_star = M.parameter('w_star', subspace_n)
        x = M.parameter('x', in_n)
        w = M.variable('w', subspace_n)

        dist = Expr.mul(U.T, w)

        # Constraints
        M.constraint(dist, Domain.greaterThan(-self.eps))
        M.constraint(dist, Domain.lessThan(self.eps))

        M.constraint(Expr.add(x, dist), Domain.greaterThan(-0.5))
        M.constraint(Expr.add(x, dist), Domain.lessThan(0.5))

        # OBJECTIVE

        # t >= dist^T dist
        t = M.variable()
        M.constraint(Expr.vstack(1/2, t, dist), Domain.inRotatedQCone())
        obj = Expr.add(t, Expr.mul(-2, Expr.dot(w_star, Expr.mul(U @ U.T, w))))
        # MOSEK-incompatible version
        # obj = Expr.add(t, Expr.mul(-2, Expr.dot(dist_star, dist)))

        M.objective(ObjectiveSense.Minimize, obj)

        M.setSolverParam('presolveUse', 'off')
        M.setSolverParam('intpntBasis', 'never')

        return M
