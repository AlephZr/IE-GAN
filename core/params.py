import os
import torch
import math
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class Parameters:
    def __init__(self, parser):
        """Parameter class stores all parameters for policy gradient

        Parameters:
            None

        Returns:
            None
        """

        # Env args
        self.eval_criteria = vars(parser.parse_args())['eval_criteria']
        self.dataset_name = vars(parser.parse_args())['dataset_name']
        self.download_root = vars(parser.parse_args())['download_root']

        self.total_steps = int(vars(parser.parse_args())['total_iterations'] * 1000)
        self.savetag = vars(parser.parse_args())['savetag']
        self.aux_folder = vars(parser.parse_args())['aux_folder']
        self.seed = vars(parser.parse_args())['seed']
        self.batch_size = vars(parser.parse_args())['batchsize']
        eval_size = vars(parser.parse_args())['evalsize']
        self.D_iters = vars(parser.parse_args())['D_iters']

        self.netD = vars(parser.parse_args())['netD']
        self.netG = vars(parser.parse_args())['netG']
        self.z_dim = vars(parser.parse_args())['z_dim']
        self.ngf = vars(parser.parse_args())['ngf']
        self.ndf = vars(parser.parse_args())['ndf']
        self.input_nc = vars(parser.parse_args())['input_nc']
        self.discriminator_lr = vars(parser.parse_args())['discriminator_lr']
        self.generator_lr = vars(parser.parse_args())['generator_lr']
        self.discriminator_beta1 = vars(parser.parse_args())['discriminator_beta1']
        self.discriminator_beta2 = vars(parser.parse_args())['discriminator_beta2']
        self.generator_beta1 = vars(parser.parse_args())['generator_beta1']
        self.generator_beta2 = vars(parser.parse_args())['generator_beta2']
        self.use_gp = vars(parser.parse_args())['use_gp']

        self.load_size = vars(parser.parse_args())['load_size']
        self.crop_size = vars(parser.parse_args())['crop_size']
        self.num_threads = vars(parser.parse_args())['num_threads']
        self.preprocess = vars(parser.parse_args())['preprocess']
        self.no_flip = vars(parser.parse_args())['no_flip']

        self.pop_size = vars(parser.parse_args())['popsize']
        self.crossover_size = vars(parser.parse_args())['crosssize']
        self.d_loss_mode = vars(parser.parse_args())['d_loss_mode']
        self.g_loss_mode = vars(parser.parse_args())['g_loss_mode']
        self.use_pytorch_scores = vars(parser.parse_args())['use_pytorch_scores']
        self.test_name = vars(parser.parse_args())['test_name']
        self.test_size = vars(parser.parse_args())['test_size']
        self.fid_batch_size = vars(parser.parse_args())['fid_batch_size']
        self.test_frequency = vars(parser.parse_args())['test_frequency']
        self.eval_size = max(math.ceil((self.batch_size * self.D_iters) / self.pop_size), eval_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.mixup = vars(parser.parse_args())['mixup']
        self.lambda_f = vars(parser.parse_args())['lambda_f']
        self.lambda_c = vars(parser.parse_args())['lambda_c']

        # Set seeds
        if self.seed is None:
            self.seed = random.randint(1, 10000)
        print("Random Seed: ", self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Save Results
        if self.aux_folder == '':
            self.savefolder = 'Results/Plots/' + self.aux_folder
        else:
            self.savefolder = 'Results/Plots/'
        if not os.path.exists(self.savefolder):
            os.makedirs(self.savefolder)
        # self.aux_folder = 'Results/Auxiliary/'
        # if not os.path.exists(self.aux_folder):
        #     os.makedirs(self.aux_folder)

        self.savetag += 'dataset:' + str(self.dataset_name)
        self.savetag += '_pop' + str(self.pop_size)
        self.savetag += '_crossover' + str(self.crossover_size)
        self.savetag += '_seed' + str(self.seed)
        self.savetag += '_netD' + str(self.netD)
        self.savetag += '_netG' + str(self.netG)
        self.savetag += '_g_loss_mode' + str(self.g_loss_mode)
        self.savetag += '_d_loss_mode' + str(self.d_loss_mode)

        if self.aux_folder == '':
            self.writer = SummaryWriter(log_dir='Results/tensorboard/' + self.savetag)
        else:
            self.writer = SummaryWriter(log_dir='Results/tensorboard/' + self.aux_folder + '/' + self.savetag)
