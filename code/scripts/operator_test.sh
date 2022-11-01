set -ex
CUDA_VISIBLE_DEVICES=1 python train.py --eval_criteria operator_test \
       --dataset_name CIFAR10 \
       --batchsize 32 \
       --netD EGAN32 --netG EGAN32 --ngf 128 --ndf 128 \
       --discriminator_lr 0.0002 --generator_lr 0.0002 \
       --z_dim 100 \
       --crop_size 32 --load_size 32 \
       --popsize 1 --crosssize 0 \
       --d_loss_mode vanilla --g_loss_mode vanilla \
       --D_iters 3 --use_gp \
       --test_name  --test_size 50000 --fid_batch_size 500 --test_frequency 5000 \
       --total_iterations 100 \
       --savetag GAN-minimax-GP --aux_folder GAN-minimax-GP \

