set -ex
python train.py --eval_criteria IE-GAN \
       --dataset_name MNIST \
       --batchsize 32 --evalsize 256 \
       --netD DCGAN28 --netG DCGAN28 --ngf 64 --ndf 64 \
       --discriminator_lr 0.0002 --generator_lr 0.0002 \
       --z_dim 100 \
       --crop_size 28 --load_size 28 --input_nc 1 \
       --popsize 1 --crosssize 1 \
       --d_loss_mode vanilla --g_loss_mode nsgan lsgan vanilla \
       --D_iters 1 --lambda_f 0.05 --lambda_c 0.001 \
       --test_name --test_size 50000 --fid_batch_size 500 --test_frequency 5000 \
       --total_iterations 100 \
       --savetag MNIST --aux_folder MNIST \
