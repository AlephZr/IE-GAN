# import time
import tflib as lib
import tflib.save_images
# import tflib.plot
# import tflib.inception_score
# import core.buffer

# import numpy as np

import torch
# import torchvision
from torch import nn, autograd
from torch.optim import Adam
import copy

import os
import sys

sys.path.append(os.getcwd())


# Download CIFAR-10 (Python version) at
# https://www.cs.toronto.edu/~kriz/cifar.html and fill in the path to the
# extracted files here!


class GAN(object):
    def __init__(self, args, model_constructor, env_constructor):
        self.args = args
        self.env = env_constructor

        self.discriminator_lr = args.discriminator_lr
        self.generator_lr = args.generator_lr
        self.discriminator_beta1 = args.discriminator_beta1
        self.discriminator_beta2 = args.discriminator_beta2
        self.generator_beta1 = args.generator_beta1
        self.generator_beta2 = args.generator_beta2
        self.batch_size = args.batch_size
        self.input_dim = args.z_dim
        self.device = args.device

        self.LAMBDA = 10  # Gradient penalty lambda hyperparameter
        self.CRITIC_ITERS = args.D_iters  # How many critic iterations per generator iteration

        self.writer = args.writer

        # Loss function
        self.criterion_BCEL = nn.BCEWithLogitsLoss()
        self.criterion_MSE = nn.MSELoss()
        # Adversarial ground truths
        self.ones_label = torch.ones(self.batch_size, device=self.device)
        self.zeros_label = torch.zeros(self.batch_size, device=self.device)

        self.netG = model_constructor.make_model('Generator')
        self.optimizerG = Adam(self.netG.parameters(), lr=self.generator_lr,
                               betas=(self.generator_beta1, self.generator_beta2))
        self.netD = model_constructor.make_model('Discriminator')
        self.optimizerD = Adam(self.netD.parameters(), lr=self.discriminator_lr,
                               betas=(self.discriminator_beta1, self.discriminator_beta2))

        print(self.netG)
        print(self.netD)

        self.one = torch.FloatTensor([1])
        self.one = self.one.to(device=self.device)
        self.mone = self.one * -1
        self.mone = self.mone.to(device=self.device)

        self.num_updates = 0

        if self.args.d_loss_mode == 'vanilla' or self.args.d_loss_mode == 'nsgan':
            self.update_D_parameters = self.update_vanilla_nsgan
            # self.update_D_parameters = self.update_mixup
        elif self.args.d_loss_mode == 'lsgan':
            self.update_D_parameters = self.update_lsgan
        elif self.args.d_loss_mode == 'wgan':
            self.update_D_parameters = self.update_wgan
        elif self.args.d_loss_mode == 'rsgan':
            self.update_D_parameters = self.update_rsgan
        else:
            raise NotImplementedError('gan mode %s not implemented' % self.args.d_loss_mode)

    def calc_gradient_penalty(self, discriminator, real_data, fake_data):
        # print("real_data: ", real_data.size(), fake_data.size())
        alpha = torch.rand(self.batch_size, 1)
        if self.args.dataset_name != '8gaussians' and self.args.dataset_name != '25gaussians' and self.args.dataset_name != 'swissroll':
            alpha = alpha.expand(self.batch_size, real_data.nelement() // self.batch_size).contiguous().view(
                self.batch_size, self.args.input_nc, self.args.crop_size, self.args.crop_size)
        else:
            alpha = alpha.expand(self.batch_size, real_data.nelement() // self.batch_size).contiguous().view(
                self.batch_size, 2)
        alpha = alpha.to(device=self.device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.to(device=self.device)
        interpolates = interpolates.requires_grad_(True)

        disc_interpolates = discriminator(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(
                                      disc_interpolates.size()).to(device=self.device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.LAMBDA
        return gradient_penalty

    def update_vanilla_nsgan(self, gen_images, real_images):
        ############################
        # (1) Update D network
        ###########################
        if self.netG.hasTanh:
            gen_images = gen_images.tanh()

        for p in self.netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update
        for i in range(self.CRITIC_ITERS):
            self.netD.zero_grad()

            ### vanilla / nsgan
            # train with real
            real_batch = real_images[i * self.batch_size:(i + 1) * self.batch_size].detach()
            output = self.netD(real_batch)
            errD_real = self.criterion_BCEL(output, self.ones_label)
            errD_real.backward()
            # train with fake
            gen_batch = gen_images[i * self.batch_size:(i + 1) * self.batch_size].detach()
            output = self.netD(gen_batch)
            errD_fake = self.criterion_BCEL(output, self.zeros_label)
            errD_fake.backward()
            # train with gradient penalty
            if self.args.use_gp:
                gradient_penalty = self.calc_gradient_penalty(self.netD, real_batch.detach(), gen_batch.detach())
                gradient_penalty.backward()
            else:
                gradient_penalty = 0.
            errD = errD_real + errD_fake + gradient_penalty
            self.optimizerD.step()

        self.writer.add_scalar('train disc cost', errD.item(), self.num_updates)
        self.num_updates += 1

    def update_lsgan(self, gen_images, real_images):
        ############################
        # (1) Update D network
        ###########################
        for p in self.netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update
        for i in range(self.CRITIC_ITERS):
            self.netD.zero_grad()

            ### lsgan
            # train with real
            real_batch = real_images[i * self.batch_size:(i + 1) * self.batch_size].detach()
            output = self.netD(real_batch)
            errD_real = self.criterion_MSE(output, self.ones_label)
            errD_real.backward()
            # train with fake
            gen_batch = gen_images[i * self.batch_size:(i + 1) * self.batch_size].detach()
            output = self.netD(gen_batch)
            errD_fake = self.criterion_MSE(output, self.zeros_label)
            errD_fake.backward()
            # train with gradient penalty
            if self.args.use_gp:
                gradient_penalty = self.calc_gradient_penalty(self.netD, real_batch.detach(), gen_batch.detach())
                gradient_penalty.backward()
            else:
                gradient_penalty = 0.
            errD = errD_real + errD_fake + gradient_penalty
            self.optimizerD.step()

        self.writer.add_scalar('train disc cost', errD.item(), self.num_updates)
        self.num_updates += 1

    def update_wgan(self, gen_images, real_images):
        ############################
        # (1) Update D network
        ###########################
        for p in self.netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update
        for i in range(self.CRITIC_ITERS):
            self.netD.zero_grad()

            ### WGAN-GP
            # train with real
            real_batch = real_images[i * self.batch_size:(i + 1) * self.batch_size].detach()
            output = self.netD(real_batch)
            errD_real = output.mean()
            errD_real.backward(self.mone)
            # train with fake
            gen_batch = gen_images[i * self.batch_size:(i + 1) * self.batch_size].detach()
            output = self.netD(gen_batch)
            errD_fake = output.mean()
            errD_fake.backward(self.one)
            # train with gradient penalty
            gradient_penalty = self.calc_gradient_penalty(self.netD, real_batch.detach(), gen_batch.detach())
            gradient_penalty.backward()
            errD = errD_fake - errD_real + gradient_penalty
            Wasserstein_D = errD_real - errD_fake

            self.optimizerD.step()
        self.writer.add_scalar('train disc cost', errD.item(), self.num_updates)
        self.writer.add_scalar('wasserstein distance', Wasserstein_D.item(), self.num_updates)
        self.num_updates += 1

    def update_rsgan(self, gen_images, real_images):
        ############################
        # (1) Update D network
        ###########################
        for p in self.netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update
        for i in range(self.CRITIC_ITERS):
            self.netD.zero_grad()

            ### rsgan
            real_batch = real_images[i * self.batch_size:(i + 1) * self.batch_size].detach()
            real_output = self.netD(real_batch)
            # train with fake
            gen_batch = gen_images[i * self.batch_size:(i + 1) * self.batch_size].detach()
            gen_output = self.netD(gen_batch)
            errD = self.criterion_BCEL(real_output - gen_output, self.ones_label)
            errD.backward()
            self.optimizerD.step()

        self.writer.add_scalar('train disc cost', errD.item(), self.num_updates)
        self.num_updates += 1

    # def mixup_batch(self, mixup=0.0, real=None, fake=None):
    #     def one_batch():
    #         data = torch.cat((real, fake))
    #         ones = torch.ones(real.size(0), device=self.device)
    #         zeros = torch.zeros(fake.size(0), device=self.device)
    #         perm = torch.randperm(data.size(0), device=self.device)
    #         labels = torch.cat((ones, zeros))
    #         return data[perm], labels[perm]
    #
    #     d1, l1 = one_batch()
    #     if mixup == 0:
    #         return d1, l1
    #     d2, l2 = one_batch()
    #     alpha = torch.randn(d1.size(0), 1, 1, 1, device=self.device).uniform_(0, mixup)
    #     d = alpha * d1 + (1. - alpha) * d2
    #     alpha = alpha.squeeze()
    #     l = alpha * l1 + (1. - alpha) * l2
    #     return d, l
    #
    # def update_mixup(self, gen_images, real_images):
    #     ############################
    #     # (1) Update D network
    #     ###########################
    #     for p in self.netD.parameters():  # reset requires_grad
    #         p.requires_grad = True  # they are set to False below in netG update
    #     for i in range(self.CRITIC_ITERS):
    #         self.netD.zero_grad()
    #
    #         ### vanilla / nsgan
    #         # train with real
    #         real_batch = real_images[i * self.batch_size:(i + 1) * self.batch_size].detach()
    #         errD_real = self.criterion_BCEL(self.netD(real_batch), self.ones_label)
    #         errD_real.backward()
    #         # train with fake
    #         gen_batch = gen_images[i * self.batch_size:(i + 1) * self.batch_size].detach()
    #         errD_fake = self.criterion_BCEL(self.netD(gen_batch), self.zeros_label)
    #         errD_fake.backward()
    #         # train with mixup
    #         data, labels = self.mixup_batch(self.args.mixup, real_batch, gen_batch)
    #         errD_mixup = self.criterion_BCEL(self.netD(data), labels)
    #         errD_mixup.backward()
    #         # train with gradient penalty
    #         if self.args.use_gp:
    #             gradient_penalty = self.calc_gradient_penalty(self.netD, real_batch.detach(), gen_batch.detach())
    #             gradient_penalty.backward()
    #         else:
    #             gradient_penalty = 0.
    #         errD = errD_real + errD_fake + errD_mixup + gradient_penalty
    #         # errD = errD_mixup + gradient_penalty
    #         self.optimizerD.step()
    #
    #     self.writer.add_scalar('train disc cost', errD.item(), self.num_updates)
    #     self.num_updates += 1
