set -ex
python train.py --eval_criteria operator_test \
       --dataset_name CIFAR10 \
       --batchsize 64 \
       --netD DCGAN32 --netG DCGAN32 --ngf 64 --ndf 64 \
       --discriminator_lr 0.0002 --generator_lr 0.0002 \
       --z_dim 100 \
       --crop_size 32 --load_size 32 \
       --popsize 1 --crosssize 0 \
       --d_loss_mode vanilla --g_loss_mode vanilla \
       --D_iters 1 \
       --test_name FID --test_size 50000 --fid_batch_size 500 --test_frequency 5000 \
       --total_iterations 100 \
       --savetag GAN --aux_folder GAN \

