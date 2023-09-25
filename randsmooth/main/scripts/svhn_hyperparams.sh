rm -rf ../../runs/svhn/rs4al1sweep
rm -rf ../../runs/svhn/rs4alinfsweep
rm -rf ../../runs/svhn/ancersweep
rm -rf ../../runs/svhn/pcasweep

rm -rf ../../out/svhn
python cert.py --train_base --data=svhn --sample_n=1000 --setup=rs4a_l1_sweep --run_cert --cert_n=500 --no_tensorboard
cp -R ../../out/svhn ../../runs/svhn/rs4al1sweep

rm -rf ../../out/svhn
python cert.py --train_base --data=svhn --sample_n=1000 --setup=rs4a_linf_sweep --run_cert --cert_n=500 --no_tensorboard 
cp -R ../../out/svhn ../../runs/svhn/rs4alinfsweep

rm -rf ../../out/svhn
python cert.py --train_base --data=svhn --sample_n=1000 --setup=ancer_sweep --run_cert --cert_n=100 --no_tensorboard 
cp -R ../../out/svhn ../../runs/svhn/ancersweep

rm -rf ../../out/svhn
python cert.py --train_base --data=svhn --sample_n=100000 --setup=project_sweep --run_cert --cert_n=500 --no_tensorboard 
cp -R ../../out/svhn ../../runs/svhn/pcasweep
