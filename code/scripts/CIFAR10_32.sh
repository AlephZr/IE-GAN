set -ex
CUDA_VISIBLE_DEVICES=1 python train.py --eval_criteria IE-GAN \
       --dataset_name CIFAR10 \
       --batchsize 32 --evalsize 256 \
       --netD EGAN32 --netG EGAN32 --ngf 128 --ndf 128 \
       --discriminator_lr 0.0002 --generator_lr 0.0002 \
       --z_dim 100 \
       --crop_size 32 --load_size 32 \
       --popsize 1 --crosssize 1 \
       --d_loss_mode vanilla --g_loss_mode nsgan lsgan vanilla \
       --D_iters 3 --lambda_f 0.05 --lambda_c 0.001 --use_gp \
       --test_name FID --test_size 50000 --fid_batch_size 100 --test_frequency 5000 \
       --total_iterations 100 \
       --savetag CIFAR10_32_GP --aux_folder CIFAR10_32_GP \
