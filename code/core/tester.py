import numpy as np
import torch
import tensorflow as tf
from collections import OrderedDict

from inception_pytorch.inception_utils import prepare_inception_metrics
from tflib.inception_score import get_inception_score
from tflib import fid


class Tester:
    def __init__(self, opt):
        self.opt = opt
        # Get the requisite network
        self.use_pytorch = self.opt.use_pytorch_scores
        self.net_type = self.opt.netG
        self.device = self.opt.device
        self.input_dim = self.opt.z_dim
        self.input_nc = self.opt.input_nc
        self.crop_size = self.opt.crop_size
        self.evaluation_size = self.opt.test_size
        self.fid_batch_size = self.opt.fid_batch_size

        self.no_FID = True
        self.no_IS = True
        self.sess, self.mu_real, self.sigma_real, self.get_inception_metrics = None, None, None, None
        if opt.use_pytorch_scores and opt.test_name:
            parallel = False
            if 'FID' in opt.test_name:
                self.no_FID = False
            if 'IS' in opt.test_name:
                self.no_IS = False
            self.get_inception_metrics = prepare_inception_metrics(opt.dataset_name, parallel, self.no_IS, self.no_FID)
        else:
            if 'FID' in opt.test_name:
                STAT_FILE = None
                if opt.dataset_name == 'CIFAR10':
                    STAT_FILE = '../metric/tflib/TTUR/stats/fid_stats_cifar10_train.npz'
                elif opt.dataset_name == 'CelebA':
                    STAT_FILE = '../metric/tflib/TTUR/stats/fid_stats_celeba.npz'
                elif opt.dataset_name == 'LSUN':
                    STAT_FILE = '../metric/tflib/TTUR/stats/fid_stats_lsun_train.npz'
                INCEPTION_PATH = '../metric/tflib/IS/imagenet'

                print("load train stats.. ")
                # load precalculated training set statistics
                f = np.load(STAT_FILE)
                self.mu_real, self.sigma_real = f['mu'][:], f['sigma'][:]
                f.close()
                print("ok")

                inception_path = fid.check_or_download_inception(INCEPTION_PATH)  # download inception network
                fid.create_inception_graph(inception_path)  # load the graph into the current TF graph

                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                self.sess = tf.Session(config=config)
                self.sess.run(tf.compat.v1.global_variables_initializer())

                self.no_FID = False
            if 'IS' in opt.test_name:
                self.no_IS = False

    # evaluate
    @torch.no_grad()
    def __call__(self, model_bucket):
        net = model_bucket

        scores_ret = OrderedDict()

        samples = torch.zeros((self.evaluation_size, 3, self.crop_size, self.crop_size), device=self.device)
        n_fid_batches = self.evaluation_size // self.fid_batch_size

        for i in range(n_fid_batches):
            frm = i * self.fid_batch_size
            to = frm + self.fid_batch_size

            if 'DCGAN' in self.net_type:
                z = torch.randn(self.fid_batch_size, self.input_dim, 1, 1, device=self.device)
            elif 'EGAN' in self.net_type:
                z = torch.rand(self.fid_batch_size, self.input_dim, 1, 1, device=self.device) * 2. - 1.
            elif self.net_type == 'WGAN':
                z = torch.randn(self.fid_batch_size, self.input_dim, device=self.device)
            else:
                raise NotImplementedError('netG [%s] is not found' % self.net_type)

            gen_s = net(z).tanh().detach()
            samples[frm:to] = gen_s
            print("\rgenerate fid sample batch %d/%d " % (i + 1, n_fid_batches), end="", flush=True)

        print("%d samples generating done" % self.evaluation_size)

        if self.use_pytorch:
            IS_mean, IS_var, FID = self.get_inception_metrics(samples, self.evaluation_size, num_splits=10)

            if not self.no_FID:
                scores_ret['FID'] = float(FID)
            if not self.no_IS:
                scores_ret['IS_mean'] = float(IS_mean)
                scores_ret['IS_var'] = float(IS_var)

        else:
            samples = samples.cpu().numpy()
            samples = ((samples + 1.0) * 127.5).astype('uint8')
            samples = samples.reshape(self.evaluation_size, self.input_nc, self.crop_size, self.crop_size)
            samples = samples.transpose(0, 2, 3, 1)

            if not self.no_FID:
                mu_gen, sigma_gen = fid.calculate_activation_statistics(samples, self.sess, batch_size=self.fid_batch_size, verbose=True)
                print("calculate FID:")
                try:
                    FID = fid.calculate_frechet_distance(mu_gen, sigma_gen, self.mu_real, self.sigma_real)
                except Exception as e:
                    print(e)
                    FID = 500
                scores_ret['FID'] = float(FID)
            if not self.no_IS:
                Imlist = []
                for i in range(len(samples)):
                    im = samples[i, :, :, :]
                    Imlist.append(im)
                print(np.array(Imlist).shape)
                IS_mean, IS_var = get_inception_score(Imlist)
                scores_ret['IS_mean'] = float(IS_mean)
                scores_ret['IS_var'] = float(IS_var)

        return scores_ret
