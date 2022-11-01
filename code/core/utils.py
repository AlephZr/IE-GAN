import numpy as np
import os
import torch
from torch import nn
from torch.nn import init
import functools
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import core.save_images

sns.set()


class Tracker:  # Tracker

    def __init__(self, save_folder, vars_string, project_string):
        self.vars_string = vars_string
        self.project_string = project_string
        self.foldername = save_folder
        self.all_tracker = [[[], 0.0, []] for _ in
                            vars_string]  # [Id of var tracked][fitnesses, avg_fitness, csv_fitnesses]
        self.counter = 0
        self.conv_size = 1
        if not os.path.exists(self.foldername):
            os.makedirs(self.foldername)

    def update(self, updates, generation):
        """Add a metric observed

        Parameters:
            updates (list): List of new scoresfor each tracked metric
            generation (int): Current gen

        Returns:
            None
        """

        self.counter += 1
        for update, var in zip(updates, self.all_tracker):
            if update is None:
                continue
            var[0].append(update)

        # Constrain size of convolution
        for var in self.all_tracker:
            if len(var[0]) > self.conv_size:
                var[0].pop(0)

        # Update new average
        for var in self.all_tracker:
            if len(var[0]) == 0: continue
            var[1] = sum(var[0]) / float(len(var[0]))

        if self.counter % 1 == 0:  # Save to csv file
            for i, var in enumerate(self.all_tracker):
                if len(var[0]) == 0:
                    continue
                var[2].append(np.array([generation, var[1]]))
                filename = self.foldername + self.vars_string[i] + self.project_string
                np.savetxt(filename, np.array(var[2]), fmt='%.3f', delimiter=',')


def pprint(l):
    """Pretty print

    Parameters:
        l (list/float/None): object to print

    Returns:
        pretty print str
    """

    if isinstance(l, list):
        if len(l) == 0:
            return None
    else:
        if l is None:
            return None
        else:
            return '%.2f' % l


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


def FC_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# For generating samples
def generate_image(frame, args, generator):
    if 'DCGAN' in args.netG:
        fixed_noise_128 = torch.randn(128, args.z_dim, 1, 1, device=args.device)
    elif 'EGAN' in args.netG:
        fixed_noise_128 = torch.rand(128, args.z_dim, 1, 1, device=args.device) * 2. - 1.
    elif args.netG == 'WGAN':
        fixed_noise_128 = torch.randn(128, args.z_dim, device=args.device)
    else:
        raise NotImplementedError('netG [%s] is not found' % args.netG)
    with torch.no_grad():
        noisev = fixed_noise_128
    samples = generator(noisev).tanh()
    samples = samples.view(-1, args.input_nc, args.crop_size, args.crop_size)
    samples = samples.mul(0.5).add(0.5)
    samples = samples.cpu().data.numpy()

    folder = args.savefolder + '/samples/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    core.save_images.save_images(samples, folder + 'samples_{}.jpg'.format(frame))


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, device, init_type='normal', init_gain=0.02):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights

    Parameters:
        device:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.

    Return an initialized network.
    """
    net.to(device)
    init_weights(net, init_type, init_gain=init_gain)
    return net


# min_max_scaler = preprocessing.MinMaxScaler()


# def dict_MinMaxScaler(dictionary):
#     if not dictionary:
#         return {}
#     else:
#         keys = []
#         values = []
#         for key, value in dictionary.items():
#             keys.append(key)
#             values.append(value)
#         values = np.array(values).reshape(-1, 1)
#         values = min_max_scaler.fit_transform(values)
#         values = values.squeeze()
#         nDict = dict(zip(keys, values))
#         return nDict


frame_index = [0]


def toy_generate_image(args, true_dist, gen_dist):
    """
    Generates and saves a plot of the true distribution, the generator, and the
    critic.
    """
    # N_POINTS = 128
    # RANGE = 3

    # points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
    # points[:, :, 0] = np.linspace(-RANGE, RANGE, N_POINTS)[:, None]
    # points[:, :, 1] = np.linspace(-RANGE, RANGE, N_POINTS)[None, :]
    # points = points.reshape((-1, 2))

    # with torch.no_grad():
    #     points_v = torch.from_numpy(points).to(device=args.device)
    # disc_map = discriminator(points_v).cpu().data.numpy()

    # with torch.no_grad():
    #     noisev = torch.randn(args.batch_size, 2, device=args.device)
    # samples = generator(noisev).cpu().data.numpy()

    plt.clf()

    # x = y = np.linspace(-RANGE, RANGE, N_POINTS)
    # plt.contour(x, y, disc_map.reshape((len(x), len(y))).transpose())

    plt.scatter(true_dist[:, 0], true_dist[:, 1], c='orange', marker='+')
    plt.scatter(gen_dist[:, 0], gen_dist[:, 1], c='green', marker='+')

    folder = folder = args.savefolder + '/samples/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(folder + 'frame_{}.jpg'.format(frame_index[0]))

    frame_index[0] += 1


kde_frame_index = [0]


def toy_generate_kde(args, generator):
    """
    Generates and saves a plot of the true distribution, the generator, and the
    critic.
    """
    with torch.no_grad():
        noisev = torch.randn(10240, args.z_dim, device=args.device)
    samples = generator(noisev).cpu().detach().numpy()

    data = {"x": samples[:, 0], "y": samples[:, 1]}
    df = pd.DataFrame(data)
    ax = sns.jointplot('x', 'y', data=df, kind='kde', color='g')
    # ax = sns.kdeplot(df.x, df.y, shade=True)

    folder = args.savefolder + '/samples/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    # ax.figure.savefig(folder + '/kde_frame_{}.png'.format(frame_index[0]))
    ax.savefig(folder + 'kde_frame_{}.png'.format(kde_frame_index[0]))

    kde_frame_index[0] += 1


def toy_true_dist_kde(args, true_dist):
    """
    Generates and saves a plot of the true distribution, the generator, and the
    critic.
    """
    true_dist = true_dist.cpu().detach().numpy()
    data = {"x": true_dist[:, 0], "y": true_dist[:, 1]}
    df = pd.DataFrame(data)
    ax = sns.jointplot('x', 'y', data=df,
                       kind='kde',
                       color='g')
    folder = args.savefolder + '/samples/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    ax.savefig(folder + 'true_dist.png')


def print_current_losses(epoch, iters, t_comp, t_data, operator):
    """print current losses on console; also save the losses to the disk

    Parameters:
        epoch (int) -- current epoch
        iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
        t_comp (float) -- computational time per data point (normalized by batch_size)
        t_data (float) -- data loading time per data point (normalized by batch_size)
    """
    message = '(epoch: %d, giters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
    message += 'selected_operator: %s' % operator
    print(message)  # print the message


def print_current_scores(epoch, iters, scores):
    """print current losses on console; also save the losses to the disk

    Parameters:
        epoch (int) -- current epoch
        iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
        scores (OrderedDict) -- training losses stored in the format of (name, float) pairs
    """
    message = '(epoch: %d, giters: %d) ' % (epoch, iters)
    for k, v in scores.items():
        message += '%s: %.3f ' % (k, v)
    print(message)  # print the message


def save_tags(args):
    file_name = 'tags.txt'
    path = os.path.join(args.savefolder, file_name)

    with open(path, 'w') as file:
        file.write('eval criteria: ' + str(args.eval_criteria) + '\n')
        file.write('dataset: ' + str(args.dataset_name) + '\n')
        file.write('pop size: ' + str(args.pop_size) + '\n')
        file.write('crossover size: ' + str(args.crossover_size) + '\n')
        file.write('seed: ' + str(args.seed) + '\n')
        file.write('netD: ' + str(args.netD) + '\n')
        file.write('netG: ' + str(args.netG) + '\n')
        file.write('g_loss_mode: ' + str(args.g_loss_mode) + '\n')
        file.write('d_loss_mode: ' + str(args.d_loss_mode) + '\n')
        file.write('lambda_f: ' + str(args.lambda_f) + '\n')
        file.write('lambda_c: ' + str(args.lambda_c) + '\n')


# def key_maximum(llist, score, device):
#     llist_T = list(zip(*llist))
#     score_T = list(zip(*score))
#     maximumlist = torch.empty(0, device=device)
#     for alist, ascore in zip(llist_T, score_T):
#         index = ascore.index(max(ascore))
#         maximumlist = torch.cat((maximumlist, alist[index].unsqueeze(0)))
#     return maximumlist


# class queue:
#     def __init__(self):
#         self.__alist = []
#
#     def push(self, value):
#         self.__alist.insert(0, value)
#
#     def pop(self):
#         return self.__alist.pop()
#
#     def size(self):
#         return len(self.__alist)
#
#     def clean(self):
#         self.__alist.clear()
#
#     def isEmpty(self):
#         return self.__alist == []
#
#     def showQueue(self):
#         print(self.__alist)
