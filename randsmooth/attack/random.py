import torch
import torch.nn as nn

from randsmooth.utils import torch_utils

import mosek
from mosek.fusion import *

from torchattacks.attack import Attack


class Random(Attack):
    # Adapted from https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/torchattacks/attacks/pgd.py

    def __init__(self, model, eps=8/255, max_perturb=True):
        super().__init__('Random', model)

        self.eps = eps
        self.max_perturb = max_perturb
        self._supported_mode = ['default']

    def forward(self, images, labels):
        assert images.shape[0] == 1

        images = images.clone().detach().to(self.device)

        delta = 2 * (torch.rand(images.shape, device=self.device) - 0.5)

        if self.max_perturb:
            delta = delta.sign()

        delta *= self.eps

        adv_images = torch.clamp(images + delta, min=0, max=1)

        return adv_images
