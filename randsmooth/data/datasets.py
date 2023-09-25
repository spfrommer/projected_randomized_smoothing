import torch
import torchvision
from torchvision import transforms
from pl_bolts.datamodules import CIFAR10DataModule

import os

from randsmooth.data.svhn_datamodule import SVHNDataModule
from randsmooth.utils import dirs, torch_utils, pretty, file_utils

from typing import List

datasets = ['cifar10', 'svhn']


def get_dataset(data):
    if data == 'cifar10':
        center = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])

        # Normalization happens as part of first layer of network
        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            center,
        ])

        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            center,
        ])

        dataset = CIFAR10DataModule(
            data_dir=dirs.data_path('cifar10'),
            batch_size=256,
            num_workers=4,
            shuffle=True,
            train_transforms=train_transforms,
            test_transforms=test_transforms,
            val_transforms=test_transforms,
        )
        dataset.in_n = 3072
    elif data == 'svhn':
        center = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])

        # Normalization happens as part of first layer of network
        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            center,
        ])

        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            center,
        ])

        dataset = SVHNDataModule(
            data_dir=dirs.data_path('svhn'),
            batch_size=256,
            num_workers=4,
            shuffle=True,
            train_transforms=train_transforms,
            test_transforms=test_transforms,
            val_transforms=test_transforms,
        )
        dataset.in_n = 3072
    else:
        raise NotImplementedError()

    dataset.prepare_data()
    dataset.setup()
    dataset.pca_components = get_pca(dataset, data)

    return dataset


def get_pca(dataset, data: str):
    file_utils.ensure_created_directory(dirs.out_path(data))
    pca_path = dirs.out_path(data, 'pca.pt')
    if not os.path.isfile(pca_path):
        pretty.subsection_print('Recomputing pca')
        pca_components = torch_utils.pca(dataset.train_dataloader())
        torch.save(pca_components, pca_path)
    else:
        pretty.subsection_print('Loading pca')
        pca_components = torch.load(pca_path)
    return pca_components


# Adapted from https://github.com/locuslab/smoothing/blob/master/code/datasets.py
class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be
      the first layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).cuda()
        self.sds = torch.tensor(sds).cuda()

    def forward(self, input: torch.Tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)

        return (input - means) / sds
