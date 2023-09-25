import torchattacks

from rs4a.src.models import WideResNet
import rs4a.src.noises as rs4a_noises


from randsmooth.base.svhnnet import SvhnNet
from randsmooth.smooth.crk_smooth import CRKSmooth
from randsmooth.smooth.project_smooth import ProjectSmooth
from randsmooth.smooth.ancer_smooth import AncerSmooth
from randsmooth.smooth.rs4a_smooth import RS4ASmooth
from randsmooth.attack.subspace import SubspacePGD
from randsmooth.attack.random import Random
from randsmooth.utils import dirs, torch_utils

from randsmooth.main.core import BaseFactory, SmoothFactory, AttackFactory


def sigma_string(sigma):
    return f'{int(100 * sigma):03d}'


def pretrain_noise_fetch(noise_type, noise_sigma, params, stability=False):
    # Ignore stability argument
    dirname = 'svhn'
    filename = f'svhn_{noise_type}_{sigma_string(noise_sigma)}.pt'
    pretrain_path = dirs.pretrain_path(dirname, filename)
    noise_classes = {'gaussian': rs4a_noises.Gaussian, 'uniform': rs4a_noises.Uniform}
    noise = noise_classes[noise_type](
        device=torch_utils.device(), dim=params.dataset.in_n, sigma=noise_sigma
    )
    return pretrain_path, noise


def main(params):
    n, sigma = params.sample_n, params.sample_sigma

    main_path, _ = pretrain_noise_fetch('gaussian', sigma, params, stability=False)

    l1_path, l1_noise = pretrain_noise_fetch('uniform', 0.25, params, stability=True)
    linf_path, linf_noise = pretrain_noise_fetch('gaussian', 0.15, params, stability=True)

    base_factories = [
        BaseFactory(
            name='base_l1', base_class=SvhnNet, epochs=0,
            params={'load_class': WideResNet, 'load_path': l1_path}
        ),
        BaseFactory(
            name='base_linf', base_class=SvhnNet, epochs=0,
            params={'load_class': WideResNet, 'load_path': linf_path}
        ),
        BaseFactory(
            name='base_reg', base_class=SvhnNet, epochs=0,
            params={'load_class': WideResNet, 'load_path': main_path}
        ),
        BaseFactory(
            name='base_project', base_class=SvhnNet, epochs=50,
            params={'load_class': WideResNet, 'load_path': main_path,
                    'project_n': 150, 'project_sigma': sigma}
        ),
    ]

    smooth_factories = [
        SmoothFactory(
            name='smooth_l1', smooth_class=RS4ASmooth, base_name='base_l1',
            params={'noise': l1_noise, 'ball_norm': 'l1', 'n0': 100, 'n': n, 'alpha': 0.001},
        ),
        SmoothFactory(
            name='smooth_linf', smooth_class=RS4ASmooth, base_name='base_linf',
            params={'noise': linf_noise, 'ball_norm': 'linf', 'n0': 100, 'n': n, 'alpha': 0.001},
        ),
        SmoothFactory(
            name='smooth_reg', smooth_class=CRKSmooth, base_name='base_reg',
            params={'sigma': sigma, 'n0': 100, 'n': n, 'alpha': 0.001},
        ),
        SmoothFactory(
            name='smooth_ancer', smooth_class=AncerSmooth, base_name='base_reg',
            params={
                'sigma': sigma, 'n0': 100, 'n': n, 'alpha': 0.001,
                'ancer_n': 100, 'ancer_iterations': 100, 'ancer_lr': 0.01,
                'ancer_kappa': 2.0, 'isotropic_iterations': 900
            },
        ),
        SmoothFactory(
            name='smooth_project', smooth_class=ProjectSmooth, base_name='base_project',
            params={'sigma': sigma, 'n0': 100, 'n': n, 'alpha': 0.001},
        ),
    ]

    return base_factories, smooth_factories


def project_sweep(params):
    n, sigma = params.sample_n, params.sample_sigma

    main_path, _ = pretrain_noise_fetch('gaussian', sigma, params, stability=False)

    base_factories, smooth_factories = [], []
    for project_n in [60, 100, 150, 200, 300, 450]:
        base_factories.append(
            BaseFactory(
                name=f'base_project{project_n}', base_class=SvhnNet, epochs=50,
                params={'load_class': WideResNet, 'load_path': main_path,
                        'project_n': project_n, 'project_sigma': sigma}
            )
        )
        smooth_factories.append(
            SmoothFactory(
                name=f'smooth_project{project_n}', smooth_class=ProjectSmooth,
                base_name=f'base_project{project_n}',
                params={'sigma': sigma, 'n0': 100, 'n': n, 'alpha': 0.001},
            ),
        )

    return base_factories, smooth_factories


def ancer_sweep(params):
    n, sigma = params.sample_n, params.sample_sigma

    main_path, _ = pretrain_noise_fetch('gaussian', sigma, params, stability=False)

    base_factories = [
        BaseFactory(
            name='base_reg', base_class=SvhnNet, epochs=0,
            params={'load_class': WideResNet, 'load_path': main_path}
        ),
    ]

    smooth_factories = []
    for lr in [0.03, 0.01, 0.003, 0.001]:
        smooth_factories.append(
            SmoothFactory(
                name=f'smooth_ancer_{lr}', smooth_class=AncerSmooth, base_name='base_reg',
                params={
                    'sigma': sigma, 'n0': 100, 'n': n, 'alpha': 0.001,
                    'ancer_n': 100, 'ancer_iterations': 100, 'ancer_lr': lr,
                    'ancer_kappa': 2.0, 'isotropic_iterations': 900
                },
            ),
        )

    return base_factories, smooth_factories


def rs4a_l1_sweep(params):
    n, sigma = params.sample_n, params.sample_sigma

    base_factories, smooth_factories = [], []

    for sigma in [0.25, 0.50, 0.75, 1.00]:
        ss = sigma_string(sigma)
        model_name = f'base_l1_{ss}'
        path, noise = pretrain_noise_fetch('uniform', sigma, params, stability=True)

        base_factories.append(
            BaseFactory(
                name=model_name, base_class=SvhnNet, epochs=0,
                params={'load_class': WideResNet, 'load_path': path}
            )
        )

        smooth_factories.append(
            SmoothFactory(
                name=f'smooth_l1_{ss}',
                smooth_class=RS4ASmooth, base_name=model_name,
                params={'noise': noise, 'ball_norm': 'l1', 'n0': 100, 'n': n, 'alpha': 0.001}
            )
        )

    return base_factories, smooth_factories


def rs4a_linf_sweep(params):
    n, sigma = params.sample_n, params.sample_sigma

    base_factories, smooth_factories = [], []

    for sigma in [0.15, 0.25, 0.50]:
        ss = sigma_string(sigma)
        model_name = f'base_linf_{ss}'
        path, noise = pretrain_noise_fetch('gaussian', sigma, params, stability=True)

        base_factories.append(
            BaseFactory(
                name=model_name, base_class=SvhnNet, epochs=0,
                params={'load_class': WideResNet, 'load_path': path}
            )
        )

        smooth_factories.append(
            SmoothFactory(
                name=f'smooth_linf_{ss}',
                smooth_class=RS4ASmooth, base_name=model_name,
                params={'noise': noise, 'ball_norm': 'linf', 'n0': 100, 'n': n, 'alpha': 0.001}
            )
        )

    return base_factories, smooth_factories


def attack(params):
    main_path, _ = pretrain_noise_fetch('gaussian', params.sample_sigma, params, stability=False)

    base_factory = BaseFactory(
        name='base', base_class=SvhnNet, epochs=0,
        params={'load_class': WideResNet, 'load_path': main_path}
    )

    attack_space = params.dataset.pca_components[59:].to(torch_utils.device()) # 95%
    # attack_space = params.dataset.pca_components[187:].to(torch_utils.device())  # 99%
    attack_factories = []
    for eps in [4 / 255, 8 / 255, 16 / 255, 32 / 255, 64 / 255]:
        eps_string = f'{int(255 * eps)}'
        attack_factories.append(
            AttackFactory(
                name=f'pgd_{eps_string}', attack_class=torchattacks.PGD,
                params={
                    'eps': eps, 'alpha': 2 / 255, 'steps': 40
                }
            )
        )
        attack_factories.append(
            AttackFactory(
                name=f'subspace_{eps_string}', attack_class=SubspacePGD,
                params={
                    'subspace': attack_space, 'grad_sign': True,
                    'eps': eps, 'alpha': eps / 4, 'steps': 5
                }
            )
        )
        attack_factories.append(
            AttackFactory(
                name=f'random_uniform_{eps_string}', attack_class=Random,
                params={
                    'eps': eps, 'max_perturb': False
                }
            )
        )
        attack_factories.append(
            AttackFactory(
                name=f'random_max_{eps_string}', attack_class=Random,
                params={
                    'eps': eps, 'max_perturb': True
                }
            )
        )

    return base_factory, attack_factories


experiments = {
    'main': main, 'project_sweep': project_sweep, 'ancer_sweep': ancer_sweep,
    'rs4a_l1_sweep': rs4a_l1_sweep, 'rs4a_linf_sweep': rs4a_linf_sweep
}
