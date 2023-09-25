rm -rf ../../runs/cifar10/rs4al1sweep
rm -rf ../../runs/cifar10/rs4alinfsweep
rm -rf ../../runs/cifar10/ancersweep
rm -rf ../../runs/cifar10/pcasweep

rm -rf ../../out/cifar10
python cert.py --train_base --data=cifar10 --sample_n=1000 --setup=rs4a_l1_sweep --run_cert --cert_n=500 --no_tensorboard
cp -R ../../out/cifar10 ../../runs/cifar10/rs4al1sweep

rm -rf ../../out/cifar10
python cert.py --train_base --data=cifar10 --sample_n=1000 --setup=rs4a_linf_sweep --run_cert --cert_n=500 --no_tensorboard 
cp -R ../../out/cifar10 ../../runs/cifar10/rs4alinfsweep

rm -rf ../../out/cifar10
python cert.py --train_base --data=cifar10 --sample_n=1000 --setup=ancer_sweep --run_cert --cert_n=100 --no_tensorboard 
cp -R ../../out/cifar10 ../../runs/cifar10/ancersweep

rm -rf ../../out/cifar10
python cert.py --train_base --data=cifar10 --sample_n=100000 --setup=project_sweep --run_cert --cert_n=500 --no_tensorboard 
cp -R ../../out/cifar10 ../../runs/cifar10/pcasweep
