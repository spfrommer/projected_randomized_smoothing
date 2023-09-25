from randsmooth.data import datasets
from randsmooth.main.cert import vols_figure, radii_figure, CertResult, setups
from randsmooth.utils import pretty, dirs
from randsmooth.main import cifar_setups, svhn_setups

import matplotlib.pyplot as plt
import numpy as np

import click
import pickle
import collections



@click.command()
def run():
    pretty.section_print('Loading certification results')
    with open(dirs.out_path('cifar10', 'cert', 'cert_results.pkl'), 'rb') as f:
        results = pickle.load(f)

    crk_results = results[0]
    proj_results = results[1]

    def robust_accuracy(results):
        return np.mean([(r.target == r.debug_vars['adv_pred']).cpu().int().item() for r in results])
    def clean_accuracy(results):
        return np.mean([(r.target == r.smooth_pred).cpu().int().item() for r in results])

    crk_robust, crk_clean = robust_accuracy(crk_results), clean_accuracy(crk_results)
    proj_robust, proj_clean = robust_accuracy(proj_results), clean_accuracy(proj_results)

    print(f'CRK robust ({crk_robust}), clean ({crk_clean})')
    print(f'Proj robust ({proj_robust}), clean ({proj_clean})')

if __name__ == "__main__":
    run()
