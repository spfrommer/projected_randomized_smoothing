from randsmooth.data import datasets
from randsmooth.main.cert import vols_figure, radii_figure, CertResult, setups
from randsmooth.utils import pretty
from randsmooth.main import cifar_setups, svhn_setups

import matplotlib.pyplot as plt

import click
import pickle
import collections


figure_types = ['cert_vols', 'cert_radii']
titles = ['ancer', 'projrs', 'l1', 'linf']


@click.command()
@click.option('--results_path', default=None)
@click.option('--figure_path', default=None)
@click.option('--data', type=click.Choice(datasets.datasets), default='cifar10')
@click.option('--setup', type=click.Choice(setups), default='main')
@click.option('--title', type=click.Choice(titles), default=None)
@click.option('--figure_type', type=click.Choice(figure_types), default='cert_sweep')
@click.option('--figsize', type=click.Choice(['1wide', '2wide', '3wide']), default='2wide')
@click.option('--cmap_shade/--cmap_default', default=False)
@click.option('--sample_n', default=1000)
@click.option('--sample_sigma', default=0.15)
def run(results_path, figure_path, data, setup, title, figure_type, figsize,
        cmap_shade, sample_n, sample_sigma):
    dataset = datasets.get_dataset(data)

    local_vars = locals()
    params = collections.namedtuple('Params', local_vars.keys())(*local_vars.values())

    pretty.section_print('Loading certification results')
    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    setups = cifar_setups if data == 'cifar10' else svhn_setups
    setups_dict = {
        'main': setups.main, 'project_sweep': setups.project_sweep,
        'ancer_sweep': setups.ancer_sweep,
        'rs4a_l1_sweep': setups.rs4a_l1_sweep, 'rs4a_linf_sweep': setups.rs4a_linf_sweep
    }

    size = {'1wide': (3.7, 2.0), '2wide': (2.6, 2.5), '3wide': (2, 1.5)}

    base_factories, smooth_factories = setups_dict[setup](params)
    if figure_type == 'cert_vols':
        title_map = {'ancer': r'ANCER', 'projrs': r'ProjectedRS$',
                     'l1': r'RS4A-l1$', 'linf': r'RS4A-linf'}
        # title_map = {'ancer': r'$\textsc{ANCER}$', 'projrs': r'$\textsc{ProjectedRS}^*$',
                     # 'l1': r'$\textsc{RS4A}-\ell_{1}$', 'linf': r'$\textsc{RS4A}-\ell_{\infty}$'}
        if title is not None:
            title = title_map[title]

        label_map = collections.OrderedDict([
            ('smooth_project', r'ProjectedRS'),
            ('smooth_reg', r'RS'),
            ('smooth_ancer', r'ANCER'),
            ('smooth_l1', r'RS4A-l1'),
            ('smooth_linf', r'RS4A-linf'),
            # ('smooth_project', r'$\textsc{ProjectedRS}^*$'),
            # ('smooth_reg', r'$\textsc{RS}$'),
            # ('smooth_ancer', r'$\textsc{ANCER}$'),
            # ('smooth_l1', r'$\textsc{RS4A}-\ell_{1}$'),
            # ('smooth_linf', r'$\textsc{RS4A}-\ell_{\infty}$'),

            ('smooth_project60', r'$p=60$'),
            ('smooth_project100', r'$p=100$'),
            ('smooth_project150', r'$p=150$'),
            ('smooth_project200', r'$p=200$'),
            ('smooth_project300', r'$p=300$'),
            ('smooth_project450', r'$p=450$'),
            ('smooth_project600', r'$p=600$'),
            ('smooth_project1000', r'$p=1000$'),

            ('smooth_ancer_0.3', r'$lr=0.03$'),
            ('smooth_ancer_0.1', r'$lr=0.01$'),
            ('smooth_ancer_0.03', r'$lr=0.03$'),
            ('smooth_ancer_0.01', r'$lr=0.01$'),
            ('smooth_ancer_0.003', r'$lr=0.003$'),
            ('smooth_ancer_0.001', r'$lr=0.001$'),

            ('smooth_l1_015', r'$\sigma=0.15$'),
            ('smooth_l1_025', r'$\sigma=0.25$'),
            ('smooth_l1_050', r'$\sigma=0.50$'),
            ('smooth_l1_075', r'$\sigma=0.75$'),
            ('smooth_l1_100', r'$\sigma=1.00$'),
            ('smooth_l1_125', r'$\sigma=1.25$'),

            ('smooth_linf_015', r'$\sigma=0.15$'),
            ('smooth_linf_025', r'$\sigma=0.25$'),
            ('smooth_linf_050', r'$\sigma=0.50$'),
            ('smooth_linf_075', r'$\sigma=0.75$'),
            ('smooth_linf_100', r'$\sigma=1.00$'),
            ('smooth_linf_125', r'$\sigma=1.25$'),
        ])
        # Technically should be passing smooths but just accessing name attribute
        fig = vols_figure(smooth_factories, results, size=size[figsize],
                          label_map=label_map, legend_title=title, cmap_shade=cmap_shade,
                          in_n=params.dataset.in_n)
    if figure_type == 'cert_radii':
        label_map = collections.OrderedDict([
            ('smooth_project', r'$\textsc{ProjectedRS}^*$'),
            ('smooth_reg', r'$\textsc{RS}$'),
            ('smooth_ancer', r'$\textsc{ANCER}$'),
        ])

        # Technically should be passing smooths but just accessing name attribute
        fig = radii_figure(smooth_factories, results, size=size[figsize], label_map=label_map)

    fig.savefig(figure_path, transparent=True, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    run()
