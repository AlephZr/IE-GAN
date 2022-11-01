import time

import numpy as np
import tensorflow as tf
import torch
import os

from algos.neuroevolution import SSNE
from core import utils
from core.tester import Tester
from inception_pytorch.inception_utils import prepare_inception_metrics
from tflib import fid
from envs_repo.MMD_computer import MMD_loss
from core.log import Visualizer


class IEGAN_Trainer:

    def __init__(self, args, model_constructor, env_constructor):
        self.args = args
        self.device = args.device
        self.env = env_constructor
        self.D_ss = args.batch_size * args.D_iters  # D's sample size
        self.rsgan_ss = self.D_ss + args.batch_size if 'rsgan' in args.g_loss_mode else self.D_ss
        self.egan_ss = self.rsgan_ss + args.eval_size if args.eval_criteria == 'E-GAN' else self.rsgan_ss

        # PG Learner
        from algos.gan import GAN
        self.learner = GAN(args, model_constructor, env_constructor)

        # Evolution
        self.evolver = SSNE(self.args, self.learner, model_constructor)

        self.tester = Tester(args)

        self.logger = Visualizer(args)

    def forward_generation(self, gen, real_samples, rsgan_samples, egan_samples):

        # # Sets the number of threads used for intraop parallelism on CPU.
        # torch.set_num_threads(4)

        # NeuroEvolution's probabilistic selection and recombination step
        gen_samples, selected_operator = self.evolver.epoch(gen, rsgan_samples, egan_samples)

        ############# UPDATE PARAMS USING GRADIENT DESCENT ##########
        self.learner.update_D_parameters(gen_samples, real_samples)

        return selected_operator

    def train(self, iterations_limit):
        # Define Tracker class to track scores
        test_tracker = utils.Tracker(self.args.savefolder, ['score'], '.csv')  # Tracker class to log progress

        if self.args.dataset_name == '8gaussians' or self.args.dataset_name == '25gaussians' or self.args.dataset_name == 'swissroll':
            mmd_computer = MMD_loss()

        time_start = time.time()

        epoch_num = 0
        gen = 1
        while gen < iterations_limit + 1:  # Infinite generations
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()  # timer for data loading per iteration

            for _, real_samples in enumerate(self.env.train_dataset):
                iter_start_time = time.time()  # timer for computation per iteration

                if self.args.dataset_name != '8gaussians' and self.args.dataset_name != '25gaussians' and self.args.dataset_name != 'swissroll':
                    real_samples = real_samples[0]
                else:
                    test_samples = real_samples[:512]
                    real_samples = real_samples[512:]

                # Train one iteration
                selected_operator = self.forward_generation(gen, real_samples[:self.D_ss].to(device=self.device),
                                                            real_samples[self.D_ss:self.rsgan_ss].to(device=self.device),
                                                            real_samples[self.rsgan_ss:self.egan_ss].to(device=self.device))

                if gen % 1000 == 0:
                    t_data = iter_start_time - iter_data_time
                    t_comp = (time.time() - iter_start_time) / self.args.batch_size
                    utils.print_current_losses(epoch_num, gen, t_comp, t_data, selected_operator)

                ###### TEST SCORE ######
                if gen % 5000 == 0:
                    self.learner.netG.load_state_dict(self.evolver.genes[-1])
                    folder = self.args.savefolder + '/checkpoints/'
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    torch.save(self.learner.netG, folder + 'netG_{}.pth'.format(gen))
                    self.logger.record_time(gen, time_start)

                if self.args.test_name and (gen % self.args.test_frequency == 0 or gen == 1):
                    self.logger.record_time(gen, time_start)
                    # FIGURE OUT THE CHAMP POLICY AND SYNC IT TO TEST
                    self.learner.netG.load_state_dict(self.evolver.genes[-1])
                    self.learner.netG.eval()

                    scores = self.tester(self.learner.netG)
                    self.logger.print_current_scores(epoch_num, gen, scores)

                    self.learner.netG.train()
                    self.logger.record_time(gen, time_start)

                if gen % 1000 == 0:
                    self.learner.netG.load_state_dict(self.evolver.genes[-1])
                    self.learner.netG.eval()

                    if self.args.dataset_name == '8gaussians' or self.args.dataset_name == '25gaussians' or self.args.dataset_name == 'swissroll':
                        # toy_true_dist_kde(self.args, real_samples)
                        utils.toy_generate_kde(self.args, self.learner.netG)

                        with torch.no_grad():
                            noisev = torch.randn(512, self.args.z_dim, device=self.args.device)
                        gen_samples = self.learner.netG(noisev).detach()
                        test_samples = test_samples.to(device=self.device)
                        mmd2 = mmd_computer.forward(gen_samples, test_samples)
                        # mmd2 = abs(compute_metric_mmd2(gen_samples, test_samples))
                        test_tracker.update([mmd2], gen)
                        self.args.writer.add_scalar('mmd2', mmd2, gen)

                        test_samples = test_samples.detach().cpu().numpy()
                        gen_samples = gen_samples.detach().cpu().numpy()
                        utils.toy_generate_image(self.args, test_samples, gen_samples)
                    else:
                        utils.generate_image(gen, self.args, self.learner.netG)

                    self.learner.netG.train()

                gen += 1

            epoch_num += 1
            print('(epoch_%d) End of giters %d / %d \t Time Taken: %d sec' % (
                epoch_num, gen, iterations_limit, time.time() - epoch_start_time))

        self.args.writer.close()
