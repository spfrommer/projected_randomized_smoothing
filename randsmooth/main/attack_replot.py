from randsmooth.data import datasets
from randsmooth.main.attack import sweep_figure, AttackResult
from randsmooth.utils import pretty
from randsmooth.main import cifar_setups, svhn_setups

import matplotlib.pyplot as plt

import click
import pickle
import collections


figure_types = ['attack_sweep']


@click.command()
@click.option('--results_path', default=None)
@click.option('--figure_path', default=None)
@click.option('--data', type=click.Choice(datasets.datasets), default='cifar10')
@click.option('--figure_type', type=click.Choice(figure_types), default='attack_sweep')
@click.option('--figsize', type=click.Choice(['2wide', '3wide']), default='2wide')
@click.option('--cmap_shade/--cmap_default', default=False)
@click.option('--sample_n', default=1000)
@click.option('--sample_sigma', default=0.15)
def run(results_path, figure_path, data, figure_type, figsize,
        cmap_shade, sample_n, sample_sigma):
    dataset = datasets.get_dataset(data)

    local_vars = locals()
    params = collections.namedtuple('Params', local_vars.keys())(*local_vars.values())

    pretty.section_print('Loading attack results')
    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    setups = cifar_setups if data == 'cifar10' else svhn_setups
    base_factory, attack_factories = setups.attack(params)
    if figure_type == 'attack_sweep':
        label_map = collections.OrderedDict([
            ('pgd', r'$\textsc{PGD}$'),
            ('subspace', r'$\textsc{SubspacePGD}$'),
            ('random_max', r'$\textsc{RandMax}$'),
            ('random_uniform', r'$\textsc{RandUniform}$'),
        ])
        # Technically should be passing smooths but just accessing name attribute
        size = {'2wide': (2.9, 2.1), '3wide': (2, 1.5)}
        fig = sweep_figure(attack_factories, results, size=size[figsize], label_map=label_map)

    fig.savefig(figure_path, transparent=True, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    run()
