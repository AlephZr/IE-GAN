import copy
import numpy as np
import torch
from torch.optim import Adam
from scipy.special import comb

from core.runner import Evaluator
from core.utils import key_maximum
from core.utils import dict_MinMaxScaler
from core import utils
import random


class SSNE:

    def __init__(self, args, learner, model_constructor):
        self.args = args
        self.device = args.device
        self.input_dim = args.z_dim

        self.g_loss_number = len(args.g_loss_mode)
        self.mutate_size = args.pop_size * self.g_loss_number
        max_crossover = int(comb(self.mutate_size, 2))
        self.crossover_size = args.crossover_size if max_crossover >= args.crossover_size else max_crossover
        print('d_loss_mode:', args.d_loss_mode, 'g_loss_mode:', args.g_loss_mode, 'mutate_size:', self.mutate_size,
              'crossover_size:', self.crossover_size)

        self.individual = learner.netG
        self.individual_optimizer = learner.optimizerG
        self.critic = learner.netD
        self.genes = []
        self.genes_optimizer = []
        for _ in range(args.pop_size):
            netG = model_constructor.make_model('Generator')
            optimizerG = Adam(netG.parameters(), lr=self.args.generator_lr,
                              betas=(self.args.generator_beta1, self.args.generator_beta2))
            self.genes.append(copy.deepcopy(netG.state_dict()))
            self.genes_optimizer.append(copy.deepcopy(optimizerG.state_dict()))
        self.env = Evaluator(args, learner.netG, learner.netD)

        self.writer = args.writer
        # GAN TRACKERS
        self.selection_stats = {'total': 0, 'crossover': 0, 'parents': 0}
        for g_loss in args.g_loss_mode:
            self.selection_stats[g_loss] = 0
        self.f_stats = {'selected': 0, 'crossover': 0}
        for g_loss in args.g_loss_mode:
            self.f_stats[g_loss] = 0
        self.fd_stats = {'selected': 0, 'crossover': 0}
        for g_loss in args.g_loss_mode:
            self.fd_stats[g_loss] = 0
        self.fq_stats = {'selected': 0, 'crossover': 0}
        for g_loss in args.g_loss_mode:
            self.fq_stats[g_loss] = 0

        # Adversarial ground truths
        self.ones_label = torch.ones(args.batch_size, device=self.device)
        self.zeros_label = torch.zeros(args.batch_size, device=self.device)
        # Loss function
        self.BCEL_loss = torch.nn.BCEWithLogitsLoss()
        self.MSE_loss = torch.nn.MSELoss()

        # self.fitness_tracker = utils.Tracker(args.savefolder, ['fitness'], '.csv')

    def distilation_crossover(self, noise, gene1, gene1_optim, gene1_critic, gene1_sample, gene2_critic, gene2_sample,
                              offspring, offspring_optim):
        for p in self.critic.parameters():
            p.requires_grad = False  # to avoid computation

        self.individual.load_state_dict(gene1)
        self.individual_optimizer.load_state_dict(gene1_optim)

        self.individual_optimizer.zero_grad()

        eps = 0.0
        fake_batch = torch.cat((gene1_sample[gene1_critic - gene2_critic > eps],
                                gene2_sample[gene2_critic - gene1_critic >= eps])).detach()
        noise_batch = torch.cat((noise[gene1_critic - gene2_critic > eps], noise[gene2_critic - gene1_critic >= eps]))
        offspring_batch = self.individual(noise_batch)

        # Offspring Update
        policy_loss = self.MSE_loss(offspring_batch, fake_batch)
        policy_loss.backward()
        self.individual_optimizer.step()

        offspring.append(copy.deepcopy(self.individual.state_dict()))
        offspring_optim.append(copy.deepcopy(self.individual_optimizer.state_dict()))

    # def distilation_crossover_size(self, noise, gene1, gene1_optim, gene1_critic, gene1_sample, gene2_critic,
    #                                gene2_sample, offspring, offspring_optim):
    #     cs = 128
    #     noise = noise[:cs]
    #     gene1_sample = gene1_sample[:cs]
    #     gene2_sample = gene2_sample[:cs]
    #     gene1_critic = gene1_critic[:cs]
    #     gene2_critic = gene2_critic[:cs]
    #
    #     for p in self.critic.parameters():
    #         p.requires_grad = False  # to avoid computation
    #
    #     self.individual.load_state_dict(gene1)
    #     self.individual_optimizer.load_state_dict(gene1_optim)
    #
    #     self.individual_optimizer.zero_grad()
    #
    #     eps = 0.0
    #     fake_batch = torch.cat((gene1_sample[gene1_critic - gene2_critic > eps],
    #                             gene2_sample[gene2_critic - gene1_critic >= eps])).detach()
    #     noise_batch = torch.cat((noise[gene1_critic - gene2_critic > eps], noise[gene2_critic - gene1_critic >= eps]))
    #     offspring_batch = self.individual(noise_batch)
    #
    #     # Offspring Update
    #     policy_loss = self.MSE_loss(offspring_batch, fake_batch)
    #     policy_loss.backward()
    #     self.individual_optimizer.step()
    #
    #     offspring.append(copy.deepcopy(self.individual.state_dict()))
    #     offspring_optim.append(copy.deepcopy(self.individual_optimizer.state_dict()))

    # def distilation_crossover(self, noise, gene, gene_optim, critics, samples, offspring, offspring_optim):
    #     for p in self.critic.parameters():
    #         p.requires_grad = False  # to avoid computation
    #
    #     self.individual.load_state_dict(gene)
    #     self.individual_optimizer.load_state_dict(gene_optim)
    #
    #     self.individual_optimizer.zero_grad()
    #
    #     fake_batch = key_maximum(samples, critics, self.device).detach()
    #     # print(fake_batch.shape)
    #     offspring_batch = self.individual(noise)
    #     # print(offspring_fake.shape,fake_batch.shape)
    #
    #     # Offspring Update
    #     # policy_loss = self.MSE_loss_sum(offspring_batch, fake_batch) + torch.mean(offspring_batch ** 2)
    #     policy_loss = self.MSE_loss(offspring_batch, fake_batch)
    #     policy_loss.backward()
    #     self.individual_optimizer.step()
    #
    #     offspring.append(copy.deepcopy(self.individual.state_dict()))
    #     offspring_optim.append(copy.deepcopy(self.individual_optimizer.state_dict()))

    def gradient_mutate(self, mutate_pop, mutate_optim, real_samples, gene, optimizer, mode):
        batch_size = self.args.batch_size
        for p in self.critic.parameters():
            p.requires_grad = False  # to avoid computation

        if 'DCGAN' in self.args.netG:
            noise = torch.randn(batch_size, self.input_dim, 1, 1, device=self.device)
        elif self.args.netG == 'WGAN':
            noise = torch.randn(batch_size, self.input_dim, device=self.device)
        elif self.args.netG == 'FC2':
            noise = torch.randn(batch_size, self.input_dim, device=self.device)
        elif 'EGAN' in self.args.netG:
            noise = torch.rand(batch_size, self.input_dim, 1, 1, device=self.device) * 2. - 1.
        else:
            raise NotImplementedError('netG [%s] is not found' % self.args.netG)

        # Variation
        for g_loss in mode:
            self.individual.load_state_dict(gene)
            self.individual_optimizer.load_state_dict(optimizer)
            self.individual_optimizer.zero_grad()

            gen_samples = self.individual(noise)
            if self.individual.hasTanh:
                gen_samples = gen_samples.tanh()
            gen_critic = self.critic(gen_samples)
            if g_loss == 'nsgan':  # nsgan Variation
                gan_loss = self.BCEL_loss(gen_critic, self.ones_label)
            elif g_loss == 'vanilla':  # vanilla Variation
                gan_loss = -self.BCEL_loss(gen_critic, self.zeros_label)
            elif g_loss == 'lsgan':  # lsgan Variation
                gan_loss = self.MSE_loss(gen_critic, self.ones_label)
            elif g_loss == 'wgan':  # wgan Variation
                gan_loss = -gen_critic.mean()
            elif g_loss == 'rsgan':  # rsgan Variation
                real_critic = self.critic(real_samples)
                gan_loss = self.BCEL_loss(gen_critic - real_critic, self.ones_label)
            else:
                raise NotImplementedError('gan mode %s not implemented' % g_loss)
            gan_loss.backward()
            self.individual_optimizer.step()

            mutate_pop.append(copy.deepcopy(self.individual.state_dict()))
            mutate_optim.append(copy.deepcopy(self.individual_optimizer.state_dict()))

    @staticmethod
    def sort_groups_by_fitness(genomes, fitness):
        groups = []
        for i, first in enumerate(genomes):
            for second in genomes[i + 1:]:
                if fitness[first] < fitness[second]:
                    groups.append((second, first, fitness[first] + fitness[second]))
                else:
                    groups.append((first, second, fitness[first] + fitness[second]))
        return sorted(groups, key=lambda group: group[2], reverse=True)

    def epoch(self, gen, rsgan_train_images, egan_eval_imgaes):
        mode = self.args.g_loss_mode
        mode_number = self.g_loss_number

        if 'DCGAN' in self.args.netG:
            noise_mutate = torch.randn(self.args.eval_size, self.input_dim, 1, 1, device=self.device)
            noise_crossover = torch.randn(self.args.eval_size, self.input_dim, 1, 1, device=self.device)
        elif 'EGAN' in self.args.netG:
            noise_mutate = torch.rand(self.args.eval_size, self.input_dim, 1, 1, device=self.device) * 2. - 1.
            noise_crossover = torch.rand(self.args.eval_size, self.input_dim, 1, 1, device=self.device) * 2. - 1.
        elif self.args.netG == 'WGAN':
            noise_mutate = torch.randn(self.args.eval_size, self.input_dim, device=self.device)
            noise_crossover = torch.randn(self.args.eval_size, self.input_dim, device=self.device)
        elif self.args.netG == 'FC2':
            noise_mutate = torch.randn(self.args.eval_size, self.input_dim, device=self.device)
            noise_crossover = torch.randn(self.args.eval_size, self.input_dim, device=self.device)
        else:
            raise NotImplementedError('netG [%s] is not found' % self.args.netG)

        fitness = []
        gs_list = []  # generated samples list

        ########## MUTATION ############
        # Initialize mutate population
        mutate_pop = []
        mutate_optim = []

        # Mutate all genes in the population
        for i in range(self.args.pop_size):
            self.gradient_mutate(mutate_pop, mutate_optim, rsgan_train_images, gene=self.genes[i],
                                 optimizer=self.genes_optimizer[i], mode=mode)

        mutate_critics = []
        for i in range(self.mutate_size):
            self.individual.load_state_dict(mutate_pop[i])
            f, gen_critic, gen_images, fd, fq = self.env.eval_worker(mode[i % mode_number], noise_mutate, egan_eval_imgaes)
            mutate_critics.append(gen_critic)
            fitness.append(f)
            gs_list.append(gen_images)
            self.f_stats[mode[i % mode_number]] = f
            self.fd_stats[mode[i % mode_number]] = fd
            self.fq_stats[mode[i % mode_number]] = fq

        ########## CROSSOVER ############
        # Initialize crossover population
        crossover_pop = []
        crossover_optim = []
        # temp = []

        sorted_groups = SSNE.sort_groups_by_fitness(range(self.mutate_size), fitness)

        # Crossover for unselected genes with 100 percent probability
        for i in range(self.crossover_size):
            first, second, _ = sorted_groups[i % len(sorted_groups)]
            # if fitness[first] < fitness[second]:
            #     first, second = second, first
            self.distilation_crossover(noise_mutate, mutate_pop[first], mutate_optim[first], mutate_critics[first],
                                       gs_list[first], mutate_critics[second], gs_list[second],
                                       crossover_pop, crossover_optim)
            # self.distilation_crossover_size(noise_mutate, mutate_pop[first], mutate_optim[first], mutate_critics[first],
            #                                 gs_list[first], mutate_critics[second], gs_list[second], crossover_pop,
            #                                 crossover_optim)
            # self.distilation_crossover(noise_mutate, mutate_pop[second], mutate_optim[second], mutate_critics[first],
            #                            gs_list[first], mutate_critics[second], gs_list[second],
            #                            temp, [])

        # if self.crossover_size != 0:
        #     index = fitness.index(max(fitness))
        #     self.distilation_crossover(noise_mutate, mutate_pop[index], mutate_optim[index], mutate_critics, gs_list,
        #                                crossover_pop, crossover_optim)

            # # test
            # if random.random() < 0.01:
            #     parents1_eval = fitness[first]
            #     parents2_eval = fitness[second]
            #     # parents3_eval = fitness[third]
            #     print("parents:", first, parents1_eval, second, parents2_eval, fitness)
            #     self.fitness_tracker.update([fitness[first]], gen)
            #     self.fitness_tracker.update([fitness[second]], gen)
            #     flag = True
            # else:
            #     flag = False

        for i in range(self.crossover_size):
            self.individual.load_state_dict(crossover_pop[i])
            crossover_f, _, crossover_gen_images, crossover_fd, crossover_fq = self.env.eval_worker('crossover',
                                                                                                    noise_crossover,egan_eval_imgaes)
            fitness.append(crossover_f)
            gs_list.append(crossover_gen_images)
            self.f_stats['crossover'] = crossover_f
            self.fd_stats['crossover'] = crossover_fd
            self.fq_stats['crossover'] = crossover_fq

            # # test
            # if flag:
            #     child = crossover_f
            #     print("child:", child)
            #     self.fitness_tracker.update([child], gen)
            #     self.individual.load_state_dict(temp[0])
            #     worse_f, _, _, _, _ = self.env.eval_worker('crossover', noise_crossover)
            #     print("child_worse:", worse_f)
            #     self.fitness_tracker.update([worse_f], gen)

        ########## SELECTION ############
        top_n = np.argsort(fitness)[-self.args.pop_size:]

        # Sync evo to pop
        self.selection_stats['total'] += 1

        selected = None
        ss_list = []
        for i in range(self.args.pop_size):
            index = top_n[i]

            if index >= self.mutate_size:
                ss_list.append(gs_list[index])
                index = index - self.mutate_size
                self.genes[i] = copy.deepcopy(crossover_pop[index])
                self.genes_optimizer[i] = copy.deepcopy(crossover_optim[index])

                selected = 'crossover'
                self.selection_stats['crossover'] += 1
            else:
                ss_list.append(gs_list[index])
                self.genes[i] = copy.deepcopy(mutate_pop[index])
                self.genes_optimizer[i] = copy.deepcopy(mutate_optim[index])

                selected = mode[index % mode_number]
                self.selection_stats[selected] += 1

        selected_samples = torch.cat(ss_list, dim=0)
        shuffle_ids = torch.randperm(selected_samples.size()[0])
        disorder_samples = selected_samples[shuffle_ids]

        if self.args.eval_criteria != 'operator_test':
            # Migration Tracker
            select_rate_dict = {g_loss: self.selection_stats[g_loss] / self.selection_stats['total'] for g_loss in mode}
            select_times_dict = {g_loss: self.selection_stats[g_loss] for g_loss in mode}
            if self.crossover_size != 0:
                select_rate_dict.update(
                    {'crossover': self.selection_stats['crossover'] / self.selection_stats['total']})
                select_times_dict.update({'crossover': self.selection_stats['crossover']})
            self.writer.add_scalars('select_rate', select_rate_dict, gen)
            self.writer.add_scalars('select_times', select_times_dict, gen)

        # self.f_stats['selected'] = self.f_stats[selected]
        self.fd_stats['selected'] = self.fd_stats[selected]
        # self.fq_stats['selected'] = self.fq_stats[selected]
        # self.writer.add_scalars('f', self.f_stats, gen)
        # self.writer.add_scalars('fq', self.fq_stats, gen)
        self.writer.add_scalars('fd', self.fd_stats, gen)

        return disorder_samples, selected
