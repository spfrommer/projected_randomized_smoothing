import os
# Surpress tensorflow cuda errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import time
import random
import pickle
import math
import warnings
import click
import numpy as np
import matplotlib
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import colorama
import itertools
import tqdm
import collections

from randsmooth.data import datasets
from randsmooth.base.base import Base
from randsmooth.smooth.smooth import Smooth
from randsmooth.utils import dirs, file_utils, pretty, torch_utils, math_utils

from randsmooth.utils.torch_utils import launch_tensorboard

from randsmooth.main import core, cifar_setups, svhn_setups

from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class CertResult:
    signal: Tensor
    target: Tensor
    base_pred: Tensor
    smooth_pred: Tensor
    vol_log: float
    debug_vars: Dict[str, Any]


def cert_phase(params, bases: Dict[str, Base],
               smooths: List[Smooth]) -> List[List[CertResult]]:
    # Return from outer to inner: smooths index, data index

    file_utils.ensure_created_directory(dirs.out_path(params.data, 'cert'))
    cert_results_path = dirs.out_path(params.data, 'cert', 'cert_results.pkl')

    for base in bases.values():
        base.eval()

    if not params.run_cert:
        pretty.section_print('Loading certification results')
        with open(cert_results_path, 'rb') as f:
            return pickle.load(f)

    pretty.section_print('Certifying smoothed classifiers')
    results = [[] for _ in range(len(smooths))]  # type: List

    test_dataset = params.dataset.test_dataloader().dataset
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1)
    cert_dataloader = itertools.islice(test_dataloader, params.cert_n)

    for (signal, target) in tqdm.tqdm(cert_dataloader, total=params.cert_n):
        signal, target = signal.to(torch_utils.device()), target.to(torch_utils.device())
        for i, smooth in enumerate(smooths):
            with torch.no_grad():
                base_pred = torch.argmax(smooth.base(signal))
                t0 = time.time()
                smooth_pred, vol_log, debug_vars = smooth.certify(signal)
                debug_vars['time'] = time.time() - t0

            if smooth_pred != target:
                debug_vars['radius'] = 0
                vol_log = -math.inf

            results[i].append(CertResult(
                signal=signal, target=target, base_pred=base_pred,
                smooth_pred=smooth_pred, vol_log=vol_log, debug_vars=debug_vars
            ))

    with open(cert_results_path, 'wb') as f:
        pickle.dump(results, f)

    with open(dirs.out_path(params.data, 'cert', 'smooths.txt'), 'w') as f:
        smooth_names = ','.join([smooth.name for smooth in smooths])
        f.write(smooth_names)

    return results


def display_results(params, bases: Dict[str, Base], smooths: List[Smooth],
                    cert_results: List[List[CertResult]]) -> None:
    pretty.section_print('Writing results')

    runtime_str, acc_str, median_vol_str = '', '', ''
    for smooth, results in zip(smooths, cert_results):
        runtimes = [r.debug_vars['time'] for r in results]
        runtime_str += f'{smooth.name}, {np.mean(runtimes)}\n'

        predict_correct = [(r.target == r.smooth_pred).int().item() for r in results]
        acc_str += f'{smooth.name}, {np.mean(predict_correct)}\n'

        vol_logs_b10 = [math.log(math.e, 10) * r.vol_log for r in results if r.vol_log > -math.inf]
        median_vol_str += f'{smooth.name}, {np.median(vol_logs_b10)}\n'

    with open(dirs.out_path(params.data, 'cert', 'runtime.txt'), 'w') as f:
        f.write(runtime_str)

    with open(dirs.out_path(params.data, 'cert', 'acc.txt'), 'w') as f:
        f.write(acc_str)

    with open(dirs.out_path(params.data, 'cert', 'median_vol_log.txt'), 'w') as f:
        f.write(median_vol_str)


    pretty.section_print('Plotting results')
    tensorboard_path = dirs.out_path(params.data, 'cert', 'tensorboard')
    file_utils.create_empty_directory(tensorboard_path)
    if params.tensorboard:
        launch_tensorboard(dirs.path(tensorboard_path), 6007)
    writer = SummaryWriter(tensorboard_path)

    writer.add_figure('A. Test certified vols', vols_figure(smooths, cert_results))
    # writer.add_figure('B. Test certified radii', radii_figure(smooths, cert_results))

    writer.flush()

    figs_path = dirs.out_path(params.data, 'cert', 'figs')
    file_utils.create_empty_directory(figs_path)
    fig = vols_figure(smooths, cert_results, in_n=params.dataset.in_n)
    fig.savefig(dirs.path(figs_path, 'vols.png'), transparent=True)
    plt.close(fig)



matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


def vols_figure(smooths: List[Smooth], results: List[List[CertResult]],
                label_map=collections.OrderedDict(), legend_title='',
                cmap_shade=False, size=(2.5, 2), in_n=3072):
    fig = plt.figure(figsize=size)
    ax = plt.gca()

    if cmap_shade:
        label_smooth_strings = [lm[0] for lm in label_map.items()]
        num_models = len([s for s in smooths if s.name in label_smooth_strings])
        ax.set_prop_cycle('color', [plt.cm.get_cmap('Blues')(i)
                                    for i in np.linspace(0.3, 1.0, num_models)])
    else:
        colors = iter([plt.cm.Dark2(i) for i in range(10)])
        ax.set_prop_cycle('color', colors)

    vols_all = [r.vol_log for smooth_results in results for r in smooth_results
                if not math.isinf(r.vol_log)]


    plot_vols = np.linspace(min(vols_all), max(vols_all), num=300)
    # Normalize by unit ball and scale
    # plot_vols_scaled = math.log(math.e, 10) * (plot_vols / (-math_utils.nball_vol_log(in_n, 1)))
    print(f'Alpha: {math.log(math.e, 10) * (-math_utils.nball_vol_log(in_n, 1))}' )
    plot_vols_scaled = (plot_vols / (-math_utils.nball_vol_log(in_n, 1)))
    # plot_vols_scaled = math.log(math.e, 10) * plot_vols
    # import pdb; pdb.set_trace()
    # plot_vols_scaled = np.array([math_utils.nball_radius(in_n, vol_log) for vol_log in plot_vols])

    if len(label_map) == 0:
        for smooth, res in zip(smooths, results):
            accuracies = [certified_accuracy_vol(r, res) for r in plot_vols]
            label = smooth.name
            label = label_map[label] if label in label_map.keys() else label
            plt.plot(plot_vols_scaled, accuracies, label=smooth.name)
    else:
        for (smooth_string, label) in label_map.items():
            matching = [i for i in range(len(smooths)) if smooths[i].name == smooth_string]
            if len(matching) == 0:
                continue
            assert len(matching) == 1
            accuracies = [certified_accuracy_vol(r, results[matching[0]]) for r in plot_vols]
            plt.plot(plot_vols_scaled, accuracies, label=label)

    ax.xaxis.set_major_locator(plt.MaxNLocator(5, integer=True))
    # ax.xaxis.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation())
    #ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: rf'$10^{{{int(x)}}}$'))
    # ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        # lambda x, _: rf'$10^{{{int(x) if isinstance(x, int) else x:0.1f} \alpha}}$'))

    int_ticks = (plot_vols_scaled.max() - plot_vols_scaled.min()) > 1

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: rf'$10^{{{int(x)} \alpha}}$' if int_ticks else rf'$10^{{{x:0.1f} \alpha}}$'))

    plt.xlim([min(plot_vols_scaled), max(plot_vols_scaled)])
    plt.xlabel(r'Volume')
    plt.ylabel('Certified accuracy')
    plt.legend(loc='lower left', title=legend_title)

    return fig


def certified_accuracy_vol(vol_log: float, results: List[CertResult]):
    return float(sum([(r.target == r.smooth_pred).item() and r.vol_log > vol_log
                      for r in results])) / len(results)


def radii_figure(smooths: List[Smooth], results: List[List[CertResult]],
                 label_map=collections.OrderedDict(), size=(2.5, 2)):
    fig = plt.figure(figsize=size)

    colors = iter([plt.cm.Dark2(i) for i in range(10)])
    plt.gca().set_prop_cycle('color', colors)

    smooths, results = \
        zip(*[(s, r) for (s, r) in zip(smooths, results)
            if ('radius' in r[0].debug_vars and s.name in label_map.keys())])

    radii_all = [r.debug_vars['radius'] for smooth_results in results for r in smooth_results]

    plot_radii = np.linspace(0, max(radii_all), num=300)

    for (smooth_string, label) in label_map.items():
        matching = [i for i in range(len(smooths)) if smooths[i].name == smooth_string]
        if len(matching) == 0:
            continue
        assert len(matching) == 1
        accuracies = [certified_accuracy_radius(r, results[matching[0]]) for r in plot_radii]
        plt.plot(plot_radii, accuracies, label=label)

    plt.xlabel('Radius')
    plt.ylabel('Certified accuracy')
    plt.legend(loc='lower left')
    plt.xlim([min(plot_radii), max(plot_radii)])

    return fig


def certified_accuracy_radius(radius: float, results: List[CertResult]):
    return float(sum([(r.target == r.smooth_pred).item() and r.debug_vars['radius'] > radius
                     for r in results])) / len(results)


def radii_diff_figure(smooths: List[Smooth], results: List[List[CertResult]]):
    fig = plt.figure()

    smooths, results = \
        zip(*[(s, r) for (s, r) in zip(smooths, results) if 'radius' in r[0].debug_vars])

    radii_all = np.array([[r.debug_vars['radius'] for r in smooth_results]
                          for smooth_results in results])

    pca_res_idxs = [i for (i, smooth) in enumerate(smooths) if 'pca' in smooth.name]
    reg_res_idxs = [i for (i, smooth) in enumerate(smooths) if 'reg' in smooth.name]

    if not (len(pca_res_idxs) == len(reg_res_idxs) == 1):
        pretty.subsection_print('Could not make certified radii diff figure')
        return None

    radii_diff = radii_all[pca_res_idxs[0]] - radii_all[reg_res_idxs[0]]
    plt.hist(radii_diff, alpha=0.5)
    plt.xlabel('pca radius - reg radius')
    return fig


setups = ['main', 'project_sweep', 'ancer_sweep', 'rs4a_l1_sweep', 'rs4a_linf_sweep']


@click.command()
@click.option('--train_base/--no_train_base', default=False)
@click.option('--run_cert/--no_run_cert', default=False)
@click.option('--cert_n', default=20)
@click.option('--recompute_pca/--keep_pca', default=False)
@click.option('--sample_n', default=1000)
@click.option('--sample_sigma', default=0.15)
@click.option('--data', type=click.Choice(datasets.datasets), default='cifar10')
@click.option('--setup', type=click.Choice(setups), default='main')
@click.option('--seed', default=1)
@click.option('--tensorboard/--no_tensorboard', default=True)
def run(train_base, run_cert, cert_n, recompute_pca, sample_n, sample_sigma,
        data, setup, seed, tensorboard):
    torch_utils.seed_everything(seed)

    if recompute_pca:
        assert train_base
        if os.path.exists(dirs.out_path(data, 'pca.pt')):
            pretty.section_print('Removing cached pca (if exists)')
            os.remove(dirs.out_path(data, 'pca.pt'))
        else:
            pretty.section_print('No existing pca found, will recompute')

    colorama.init()

    pretty.section_print('Loading datasets')
    warnings.filterwarnings('ignore')
    dataset = datasets.get_dataset(data)

    pretty.section_print('Assembling parameters')
    local_vars = locals()
    params = collections.namedtuple('Params', local_vars.keys())(*local_vars.values())

    setups_lookup = {'cifar10': cifar_setups, 'svhn': svhn_setups}
    setups = setups_lookup[data]
    setups_dict = {
        'main': setups.main, 'project_sweep': setups.project_sweep,
        'ancer_sweep': setups.ancer_sweep,
        'rs4a_l1_sweep': setups.rs4a_l1_sweep, 'rs4a_linf_sweep': setups.rs4a_linf_sweep
    }

    base_factories, smooth_factories = setups_dict[setup](params)

    bases = core.base_phase(base_factories, params)
    smooths = core.smooth_phase(smooth_factories, bases, params)
    cert_results = cert_phase(params, bases, smooths)
    display_results(params, bases, smooths, cert_results)


if __name__ == "__main__":
    run()
