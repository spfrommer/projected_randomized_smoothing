rm -rf ../../out/svhn
python attack.py --train_base --data=svhn --run_attack --attack_n=100 --no_tensorboard
cp -R ../../out/svhn ../../runs/svhn/attack

python attack_replot.py --results_path=../../runs/svhn/attack/attack/attack_results.pkl --figure_path=./figs/svhn_attack_sweep.png --figure_type=attack_sweep --data=svhn
