rm -rf ../../out/cifar10
python attack.py --train_base --data=cifar10 --run_attack --attack_n=100 --no_tensorboard
cp -R ../../out/cifar10 ../../runs/cifar10/attack

python attack_replot.py --results_path=../../runs/cifar10/attack/attack/attack_results.pkl --figure_path=./figs/cifar10_attack_sweep.png --figure_type=attack_sweep --data=cifar10
