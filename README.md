# Projected Randomized Smoothing for Certified Adversarial Robustness
This code accompanies our paper:

**Projected Randomized Smoothing for Certified Adversarial Robustness**\
_Samuel Pfrommer, Brendon G. Anderson, Somayeh Sojoudi_.\
Transactions on Machine Learning Research, 2023.


## Installation
This code was tested using Python 3.10.12, and is most likely easiest to reproduce using [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/). After setting up the virtual environmnet, simply run the setup script `sh setup.sh`. This requires a working MOSEK install and license as documented [here](https://docs.mosek.com/latest/install/installation.html) -- this license is freely available for academic researchers. A latex installation is also recommended for reproducing the final plots.

## Key implementations
The implementation of the "base classifier" as described in Section 2 is contained in the "base" directory. Namely, `base/projector.py` contains a Module that implements the initial projection and reconstruction. `base/cifarnet.py` and `base/svhnnet.py` compose this with a Wide ResNet to form the base, non-smoothed classification architecture.

The various smoothing methods are contained in the `smooth` directory, with our method specifically implemented in `smooth/project_smooth.py` (see Algorithm 1). The subspace attack method described in Appendix B.1 is implemented in `attack/subspace.py`.

Hyperparameter sweeps and experimental setups are contained in `main/cifar_setups.py` and `main/svhn_setups.py`.

## Reproducing results
All results are reproduced using scripts in `main/scripts`, executed from the `main` directory. For example, the CIFAR-10 hyperparameter sweeps can be executed as `bash scripts/cifar_hyperparams.sh`. The data for the main certification plots (e.g. Figure 3a) are produced by the `***_main.sh` scripts, while the attack plots of Section 4.1 are produced by `***_attack.sh` scripts.

## Attribution
The code in `lib/rs4a` is adapted from the original [github](https://github.com/tonyduan/rs4a).
