set -ex
python train.py --eval_criteria IE-GAN \
       --dataset_name 25gaussians \
       --batchsize 32 --evalsize 256 \
       --netD FC2 --netG FC2 --ngf 512 --ndf 512 \
       --discriminator_lr 0.0001 --generator_lr 0.0001 \
       --z_dim 2 \
       --popsize 1 --crosssize 1 \
       --d_loss_mode vanilla --g_loss_mode nsgan lsgan vanilla \
       --D_iters 1 --lambda_f 1 --lambda_c 0.001 \
       --test_name  \
       --total_iterations 100 \
       --savetag lf=1 --aux_folder lf=1 \
