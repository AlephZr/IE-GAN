# -- coding:UTF-8 --
import numpy as np
import argparse
import os
import random
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data

from algos.cegan_trainer import CEGAN_Trainer
from core.params import Parameters
from envs_repo.constructor import EnvConstructor
from models.constructor import ModelConstructor

parser = argparse.ArgumentParser()

#######################  COMMANDLINE - ARGUMENTS ######################
parser.add_argument('--eval_criteria', type=str,
                    help='Evaluation criteria Choices: E-GAN | IE-GAN | operator_test',
                    default='operator_test')
parser.add_argument('--dataset_name', type=str, default='25gaussians',
                    help='name of imported dataset. CIFAR10 | CIFAR100 | CelebA | LSUN | MNIST | '
                         '8gaussians | 25gaussians | swissroll')
parser.add_argument('--download_root', type=str, default='./datasets',
                    help='root directory of dataset exist or will be saved')

parser.add_argument('--seed', type=int, help='Seed')
parser.add_argument('--savetag', type=str, help='#Tag to append to savefile', default='')
parser.add_argument('--aux_folder', type=str, help='#Tag to append to dir', default='')
parser.add_argument('--gpu_id', type=int, help='#GPU ID ', default=0)
parser.add_argument('--total_iterations', type=float, help='#Total iterations in the env in kilos ', default=100)

parser.add_argument('--netD', type=str, default='FC2',
                    help='specify discriminator architecture [EGAN32 | EGAN64 | EGAN128 | DCGAN | DCGAN28 | WGAN | FC2]')
parser.add_argument('--netG', type=str, default='FC2',
                    help='specify generator architecture [EGAN32 | EGAN64 | EGAN128 | DCGAN | DCGAN28 | WGAN | FC2]')
parser.add_argument('--z_dim', type=int, default=2, help='# of input z(noise) dims: default 100')
parser.add_argument('--ngf', type=int, default=512)
parser.add_argument('--ndf', type=int, default=512)
parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
parser.add_argument('--discriminator_lr', type=float, help='discriminator learning rate?', default=0.0001)
parser.add_argument('--generator_lr', type=float, help='generator learning rate?', default=0.0001)
parser.add_argument('--discriminator_beta1', type=float, help='beta1 of adam', default=0.5)
parser.add_argument('--discriminator_beta2', type=float, help='beta2 of adam', default=0.999)
parser.add_argument('--generator_beta1', type=float, help='beta1 of adam', default=0.5)
parser.add_argument('--generator_beta2', type=float, help='beta2 of adam', default=0.999)
parser.add_argument('--mixup', type=float, help='# of mixup', default=0.2)
parser.add_argument('--lambda_f', type=float, help='balance Fq and Fd for F', default=0.05)
parser.add_argument('--lambda_c', type=float, help='balance Fq and Fd for crossover', default=0.05)
parser.add_argument('--batchsize', type=int, help='input batch size', default=64)
parser.add_argument('--evalsize', type=int, help='eval batch size', default=256)
parser.add_argument('--D_iters', type=int, default=1, help='# of iters of D after each G updating')
parser.add_argument('--use_gp', action='store_true', default=False, help='if use gradients penalty')

parser.add_argument('--load_size', type=int, default=32, help='scale images to this size')
parser.add_argument('--crop_size', type=int, default=32, help='then crop to this size')
parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
parser.add_argument('--preprocess', type=str, default='resize_and_crop',
                    help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
parser.add_argument('--no_flip', action='store_true', default=False,
                    help='if specified, do not flip the images for data augmentation')

# ALGO SPECIFIC ARGSdiscriminator_model_bucket
parser.add_argument('--popsize', type=int, help='#Policies in the population', default=1)
parser.add_argument('--crosssize', type=int, help='#Policies in rollout size', default=0)
parser.add_argument('--d_loss_mode', type=str, default='vanilla',
                    help='lsgan | nsgan | vanilla | wgan | rsgan')
parser.add_argument('--g_loss_mode', nargs='*', default=['vanilla'],
                    help='lsgan | nsgan | vanilla | wgan | rsgan')
parser.add_argument('--use_pytorch_scores', action='store_true', default=False, help='if use pytorch version scores')
parser.add_argument('--test_name', nargs='*', help='#Type of test envs：FID | IS', default=[])
parser.add_argument('--test_size', default=50000, type=int, help='# of total sample size for socre evaluation')
parser.add_argument('--fid_batch_size', default=500, type=int, help='# fid calculate batch size')
parser.add_argument('--test_frequency', default=5000, type=int, help='#Number of gen of test')

# # Figure out GPU to use [Default is 0]
os.environ['CUDA_VISIBLE_DEVICES'] = str(vars(parser.parse_args())['gpu_id'])

#######################  Construct ARGS Class to hold all parameters ######################
args = Parameters(parser)

if args.preprocess != 'scale_width':  # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
    cudnn.benchmark = True

################################## Find and Set MDP (environment constructor) ########################
env_constructor = EnvConstructor(args)

#######################  Actor, Critic and ValueFunction Model Constructor ######################
model_constructor = ModelConstructor(args)

ai = CEGAN_Trainer(args, model_constructor, env_constructor)
ai.train(args.total_steps)
