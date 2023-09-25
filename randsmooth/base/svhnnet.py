import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR

from randsmooth.data.datasets import NormalizeLayer
from randsmooth.base.base import Base
from randsmooth.base.projector import Projector
from randsmooth.utils import torch_utils


class SvhnNet(Base):
    def __init__(self, load_class, load_path=None, project_n=None, project_sigma=None, **kwargs):
        super().__init__(**kwargs)

        modules = []

        if project_n is not None:
            self.project = Projector(project_n, project_sigma, self.data)
            modules.append(self.project)

        # Networks are pretrained on [0,1] images, must shift inputs
        modules.append(NormalizeLayer(means=[-0.5, -0.5, -0.5], sds=[1.0, 1.0, 1.0]))

        core_net = load_class(dataset='svhn', device=torch_utils.device())
        if load_path is not None:
            core_net.load_state_dict(torch.load(load_path))
        modules.append(core_net)

        self.net = nn.Sequential(*modules)

        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def extra_log(self, experiment, outputs):
        if self.project is not None:
            for i in range(1):
                x = outputs[i]['signal'][0:1]
                x_reconstruct = self.project.project_with_noise(x, training_noise=False)
                catted = torch.cat([x, x_reconstruct], dim=2)
                experiment.add_image(f'Images/{i}', self.show_img(catted[0]), self.current_epoch)

    # Unnormalize images so they show properly
    def show_img(self, img):
        return torch_utils.numpy((img + 0.5).clamp(0, 1))

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        scheduler = ExponentialLR(optimizer, gamma=0.95)
        return [optimizer], [scheduler]
