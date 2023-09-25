import os
# Surpress tensorflow cuda errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import random
import pickle
import warnings
import click
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import colorama
import itertools
import tqdm
import collections

from randsmooth.attack.subspace import SubspacePGD
from randsmooth.data import datasets
from randsmooth.base.base import Base
from randsmooth.utils import dirs, file_utils, pretty, torch_utils

from randsmooth.utils.torch_utils import launch_tensorboard

from randsmooth.main import core, cifar_setups, svhn_setups
from randsmooth.main.core import AttackFactory

from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class AttackResult:
    signal: Tensor
    target: Tensor
    pred: Tensor

    adv_signal: Tensor
    adv_pred: Tensor

    debug_vars: Dict[str, Any]


def attack_phase(params, base: Base, attacks: List[AttackFactory]) -> List[List[AttackResult]]:
    # Return from outer to inner: attacks index, data index

    file_utils.ensure_created_directory(dirs.out_path(params.data, 'attack'))
    attack_results_path = dirs.out_path(params.data, 'attack', 'attack_results.pkl')

    base.eval()

    if not params.run_attack:
        pretty.section_print('Loading attack results')
        with open(attack_results_path, 'rb') as f:
            return pickle.load(f)

    pretty.section_print('Attacking classifiers')
    results = [[] for _ in range(len(attacks))]  # type: List

    test_dataset = params.dataset.test_dataloader().dataset
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1)
    eval_dataloader = itertools.islice(test_dataloader, params.attack_n)

    for j, (signal, target) in tqdm.tqdm(enumerate(eval_dataloader), total=params.attack_n):
        signal, target = signal.to(torch_utils.device()), target.to(torch_utils.device())
        for i, attack in enumerate(attacks):
            base_attack = attack.attack_class(base, **attack.params)

            with torch.no_grad():
                pred = torch.argmax(base(signal))

            # Translates [-0.5, 0.5] to [0, 1] for torch attacks
            print(f'Attacking with {attack.name}...')
            adv_signal = base_attack(signal + 0.5, target) - 0.5
            adv_pred = base(adv_signal).argmax()

            debug_vars = {}
            if j < 10:
                debug_vars['signals'] = torch.cat([signal, adv_signal], dim=2)

            results[i].append(AttackResult(
                signal=signal, target=target, pred=pred,
                adv_signal=adv_signal, adv_pred=adv_pred, debug_vars=debug_vars
            ))

            if isinstance(base_attack, SubspacePGD):
                base_attack.problem.dispose()

    with open(attack_results_path, 'wb') as f:
        pickle.dump(results, f)

    return results

# plt.rcParams.update({'font.size': 10, 'text.usetex': True, 'font.family': 'Times New Roman'})
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

def display_results(params, base: Base,
                    attacks: List[AttackFactory], results: List[List[AttackResult]]) -> None:
    pretty.section_print('Plotting attack results')
    tensorboard_path = dirs.out_path(params.data, 'attack', 'tensorboard')
    file_utils.create_empty_directory(tensorboard_path)
    if params.tensorboard:
        launch_tensorboard(dirs.path(tensorboard_path), 6007)
    writer = SummaryWriter(tensorboard_path)

    figs_path = dirs.out_path(params.data, 'attack', 'figs')
    file_utils.create_empty_directory(figs_path)

    writer.add_figure('Sweep success', sweep_figure(attacks, results))
    fig = sweep_figure(attacks, results)
    fig.savefig(dirs.path(figs_path, 'sweep.pgf'), transparent=True)
    plt.close(fig)

    writer.flush()

    for attack, attack_results in zip(attacks, results):
        adversarial_signals_figures(attack_results, attack, writer,
                                    dirs.path(figs_path, f'{attack.name}.png'))


def sweep_figure(attacks: List[AttackFactory],
                 results: List[List[AttackResult]], size=(2.5, 2), label_map={}):
    fig, ax = plt.subplots(figsize=size)

    eps = set()
    asr = collections.defaultdict(list)

    for (attack, attack_results) in zip(attacks, results):
        attack_eps = round(attack.params['eps'] * 255)
        attack_name = attack.name[:attack.name.rfind('_')]
        asr_mean = adversary_success(attack_results)

        eps.add(attack_eps)
        asr[attack_name].append(asr_mean)

    colors = iter([plt.cm.Dark2(i) for i in range(10)])
    ax.set_prop_cycle('color', colors)

    eps = sorted(eps)
    x = list(range(len(eps)))

    # for (attack_name, asr_means) in asr.items():
    for (attack_name, label) in label_map.items():
        asr_means = asr[attack_name]

        plt.plot(x, asr_means, linestyle='--', marker='o',
                 markersize=4, label=label_map[attack_name])

    plt.xticks(x, [f'{eps}/255' for eps in eps])

    plt.xlabel(r'$\ell_{\infty}$ attack radius $\epsilon$')
    plt.ylabel(r'Attack success rate')
    plt.legend()

    return fig


def adversary_success(results: List[AttackResult]):
    successes = [float((r.target != r.adv_pred).item()) for r in results if (r.target == r.pred)]
    return np.mean(successes)


def adversarial_signals_figures(results: List[AttackResult], attack, writer, save_path=None):
    signals = [r.debug_vars['signals'] for r in results[:10]]
    all_signals = torch.cat(signals, dim=3)

    all_signals = all_signals.squeeze(0) + 0.5
    all_signals = all_signals.clamp(0, 1)
    # resize = torchvision.transforms.Resize(32 * 3 * 4)
    # signals = resize(torch_utils.numpy(signals))

    writer.add_image(f'Adversarial signals {attack.name}', all_signals)

    if save_path is not None:
        save_image(all_signals, save_path)


@click.command()
@click.option('--train_base/--no_train_base', default=False)
@click.option('--run_attack/--no_run_attack', default=False)
@click.option('--attack_n', default=20)
@click.option('--recompute_pca/--keep_pca', default=False)
@click.option('--sample_sigma', default=0.15)
@click.option('--data', type=click.Choice(datasets.datasets), default='cifar10')
@click.option('--seed', default=1)
@click.option('--tensorboard/--no_tensorboard', default=True)
def run(train_base, run_attack, attack_n,
        recompute_pca, sample_sigma, data, seed, tensorboard):
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

    setups = cifar_setups if data == 'cifar10' else svhn_setups
    base_factory, attack_factories = setups.attack(params)

    bases = core.base_phase([base_factory], params)
    attack_results = attack_phase(params, list(bases.values())[0], attack_factories)
    display_results(params, bases, attack_factories, attack_results)


if __name__ == "__main__":
    run()
