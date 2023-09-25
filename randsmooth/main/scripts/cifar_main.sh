rm -rf ../../out/cifar10
python cert.py --train_base --data=cifar10 --sample_n=100000 --run_cert --cert_n=500 --no_tensorboard 
rm -rf ../../runs/cifar10/main
cp -R ../../out/cifar10 ../../runs/cifar10/main

python cert_replot.py --results_path=../../runs/cifar10/main/cert/cert_results.pkl --figure_path=./figs/cifar10_main.png --setup=main --figure_type=cert_vols --figsize=2wide --data=cifar10
