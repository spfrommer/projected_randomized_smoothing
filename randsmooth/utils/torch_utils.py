import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from lightning_fabric.utilities.seed import seed_everything

import threading
import socket
import os
import sklearn.decomposition
import matplotlib.pyplot as plt
import sys

from randsmooth.utils import file_utils


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #seed_everything(seed, True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def gpu_n():
    return 1 if torch.cuda.is_available() else 0


def numpy(tensor):
    return tensor.detach().cpu().numpy()


def imshow(X):
    if X.shape[0] == 1:
        X = X[0]

    if torch.is_tensor(X):
        X = numpy(X)

    X = np.moveaxis(X, 0, -1)

    plt.figure()
    plt.imshow(X + 0.5)
    plt.show()


def matrix_diag(diagonal):
    # From https://github.com/pytorch/pytorch/issues/12160
    # Turns a N x m tensor into a N x m x m tensor with the values along the diagonal
    N = diagonal.shape[-1]
    shape = diagonal.shape[:-1] + (N, N)
    device, dtype = diagonal.device, diagonal.dtype
    result = torch.zeros(shape, dtype=dtype, device=device)
    indices = torch.arange(result.numel(), device=device).reshape(shape)
    indices = indices.diagonal(dim1=-2, dim2=-1)
    result.view(-1)[indices] = diagonal
    return result


def build_sequential(in_n, out_n, L, H, nonlin=nn.ReLU, final_layer='logsoftmax'):
    assert L > 1

    modules = [nn.Linear(in_n, H), nonlin()]
    for _ in range(L - 2):
        modules.append(nn.Linear(H, H))
        modules.append(nonlin())
    modules.append(nn.Linear(H, out_n))
    if final_layer == 'logsoftmax':
        modules.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*modules)


def launch_tensorboard(tensorboard_dir, port, erase=True):
    if erase:
        file_utils.create_empty_directory(tensorboard_dir)

    # Use threading so tensorboard is automatically closed on process end
    command = f'tensorboard --bind_all --port {port} '\
              f'--logdir {tensorboard_dir} > /dev/null '\
              f'--window_title {socket.gethostname()} 2>&1'
    t = threading.Thread(target=os.system, args=(command,))
    t.start()

    print(f'Launching tensorboard on http://localhost:{port}')


def pca(dataloader, k=None):
    Xs = []
    cum_size = 0
    for batch in dataloader:
        Xs.append(batch[0])
        cum_size += sys.getsizeof(batch[0])

        if cum_size > 1000000000:
            break

    X = torch.cat(Xs, dim=0)
    X = X.reshape(X.shape[0], -1)
    pca = sklearn.decomposition.PCA(n_components=k)
    pca.fit(X)

    # import pdb; pdb.set_trace()
    # pca.explained_variance_ratio_.cumsum()

    return torch.tensor(pca.components_).float()


def complete_basis(partial_basis):
    # Adapted from
    # https://stackoverflow.com/questions/69178295/complete-or-extend-orthonormal-basis-in-python

    # Partial_basis is of dimenision m, n, with m < n, returns n-m x n
    x = numpy(partial_basis.T)
    n, m = x.shape
    u, s, v = np.linalg.svd(x)
    y = u[:, m:]
    return torch.tensor(y.T).type_as(partial_basis)


class SoftmaxWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return F.softmax(self.module(x))
