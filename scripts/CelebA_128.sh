set -ex
python train.py --eval_criteria IE-GAN \
       --dataset_name CelebA \
       --batchsize 32 --evalsize 256 \
       --netD DCGAN128 --netG DCGAN128 --ngf 128 --ndf 128 \
       --discriminator_lr 0.0002 --generator_lr 0.0002 \
       --z_dim 100 \
       --crop_size 128 \
       --popsize 1 --crosssize 1 \
       --d_loss_mode vanilla --g_loss_mode nsgan lsgan vanilla \
       --D_iters 1 --lambda_f 0.05 --lambda_c 0.001 --use_gp \
       --test_name --test_size 50000 --fid_batch_size 100 --test_frequency 5000 \
       --total_iterations 100 \
       --savetag CelebA_128_GP --aux_folder CelebA_128_GP \
