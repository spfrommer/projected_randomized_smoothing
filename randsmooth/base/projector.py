import torch
import torch.nn as nn


class Projector(nn.Module):
    def __init__(self, project_n, sigma, data):
        super().__init__()

        # This is actually the transpose of the projection matrix as defined in the paper
        self.project_U = nn.Parameter(data.pca_components[:project_n],
                                      requires_grad=False)
        self.sigma = sigma

    def forward(self, x):
        return self.project_with_noise(x, training_noise=True)

    def project_with_noise(self, x, training_noise=True):
        batch_n, shape_init = x.shape[0], x.shape
        x = x.reshape(batch_n, -1)

        x = x @ self.project_U.T

        if training_noise and self.training:
            x = x + torch.zeros_like(x).normal_() * self.sigma

        x = x @ self.project_U

        return x.reshape(batch_n, *shape_init[1:])
