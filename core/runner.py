import torch
from torch import nn, autograd
import random


class Evaluator:
    def __init__(self, args, generator, discriminator):
        self.args = args
        self.device = args.device
        self.d_mode = args.d_loss_mode
        self.netS = generator  # Subject
        self.netE = discriminator  # Examiner

        # Loss function
        self.criterion_BCEL = nn.BCEWithLogitsLoss()
        self.criterion_MAE = nn.L1Loss(reduction='none')

        if args.eval_criteria == 'IE-GAN':
            self.eval_worker = self.fitness_iegan
        elif args.eval_criteria == 'E-GAN' and (args.d_loss_mode == 'vanilla' or args.d_loss_mode == 'nsgan'):
            self.eval_worker = self.fitness_egan
        elif args.eval_criteria == 'operator_test':
            self.eval_worker = self.operator_test
        else:
            raise NotImplementedError('Illegal criteria！')

    def fitness_iegan(self, operator_type, fake_noise, real_samples):
        for p in self.netE.parameters():
            p.requires_grad = False

        with torch.no_grad():
            gen_samples = self.netS(fake_noise).detach()
            if self.netS.hasTanh:
                gen_samples_act = gen_samples.tanh()
            else:
                gen_samples_act = gen_samples

        F_quailty = self.netE(gen_samples_act)

        F_diversity = torch.empty(0, device=self.device)
        comp_size = 5  # 1/3/5/30
        for i in range(comp_size):
            shuffle_ids = torch.randperm(gen_samples_act.size(0))
            disorder_samples = gen_samples_act[shuffle_ids]
            loss = self.criterion_MAE(gen_samples_act, disorder_samples)
            # loss = self.criterion_MSE(gen_samples, disorder_samples).sqrt_()
            loss_samples = loss.reshape(gen_samples_act.size(0), -1).mean(1).unsqueeze(0)
            F_diversity = torch.cat((F_diversity, loss_samples))
        F_diversity = F_diversity.mean(0)

        F_critic = (F_quailty + self.args.lambda_c * F_diversity).detach().cpu().numpy()
        # F_critic = F_quailty.detach().cpu().numpy()
        # f = F_critic.mean()
        f = (F_quailty + self.args.lambda_f * F_diversity).mean()

        return f, F_critic, gen_samples, F_diversity.mean().item(), F_quailty.mean().item()

    def operator_test(self, operator_type, fake_noise, real_samples):
        for p in self.netE.parameters():
            p.requires_grad = False

        with torch.no_grad():
            gen_samples = self.netS(fake_noise).detach()
            if self.netS.hasTanh:
                gen_samples_act = gen_samples.tanh()
            else:
                gen_samples_act = gen_samples

        F_diversity = torch.empty(0, device=self.device)
        comp_size = 5  # 1/3/5/30
        for i in range(comp_size):
            shuffle_ids = torch.randperm(gen_samples_act.size(0))
            disorder_samples = gen_samples_act[shuffle_ids]
            loss = self.criterion_MAE(gen_samples_act, disorder_samples)
            # loss = self.criterion_MSE(gen_samples, disorder_samples).sqrt_()
            loss_samples = loss.reshape(gen_samples_act.size(0), -1).mean(1).unsqueeze(0)
            F_diversity = torch.cat((F_diversity, loss_samples))
        F_diversity = F_diversity.mean(0)

        return None, None, gen_samples, F_diversity.mean().item(), None

    def fitness_egan(self, operator_type, fake_noise, real_samples):
        # Dataset iterator
        for p in self.netE.parameters():
            p.requires_grad = True

        self.netE.zero_grad()

        with torch.no_grad():
            gen_samples = self.netS(fake_noise).detach()
            if self.netS.hasTanh:
                gen_samples_act = gen_samples.tanh()
            else:
                gen_samples_act = gen_samples

        # Fiegan_q
        F_diversity = torch.empty(0, device=self.device)
        comp_size = 5  # 1/3/5/30
        for i in range(comp_size):
            shuffle_ids = torch.randperm(gen_samples_act.size(0))
            disorder_samples = gen_samples_act[shuffle_ids]
            loss = self.criterion_MAE(gen_samples_act, disorder_samples)
            # loss = self.criterion_MSE(gen_samples, disorder_samples).sqrt_()
            loss_samples = loss.reshape(gen_samples_act.size(0), -1).mean(1).unsqueeze(0)
            F_diversity = torch.cat((F_diversity, loss_samples))
        F_diversity = F_diversity.mean(0)

        ### vanilla/nsgan
        D_real = self.netE(real_samples)
        D_fake = self.netE(gen_samples_act)
        D_critic = torch.sigmoid(D_fake).detach().cpu().numpy()
        Fq = D_critic.mean()
        errD_real = self.criterion_BCEL(D_real, torch.ones(self.args.eval_size, device=self.args.device))
        errD_fake = self.criterion_BCEL(D_fake, torch.zeros(self.args.eval_size, device=self.args.device))
        errD = errD_real + errD_fake

        gradients = autograd.grad(outputs=errD, inputs=self.netE.parameters(),
                                  grad_outputs=torch.ones(errD.size()).to(device=self.device), create_graph=True,
                                  retain_graph=True, only_inputs=True)
        with torch.no_grad():
            for i, grad in enumerate(gradients):
                grad = grad.view(-1)
                allgrad = grad if i == 0 else torch.cat([allgrad, grad])
        # Fd = -torch.log(torch.norm(allgrad) + self.args.eps).detach().cpu().numpy()
        Fd = -torch.log(torch.norm(allgrad)).detach().cpu().numpy()
        fitness = Fq + self.args.lambda_f * Fd

        return fitness, D_critic, gen_samples, F_diversity.mean().item(), None
