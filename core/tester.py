from core import utils as utils
import numpy as np
import torch
from collections import OrderedDict
from tflib.inception_score import get_inception_score
from tflib import fid


# evaluate
@torch.no_grad()
def tester(args, model_bucket, test_fid, test_is, sess, mu_real, sigma_real, get_inception_metrics):
    # Get the requisite network
    use_pytorch = args.use_pytorch_scores
    net_type = args.netG
    device = args.device
    net = model_bucket
    input_dim = args.z_dim
    input_nc = args.input_nc
    crop_size = args.crop_size
    evaluation_size = args.test_size
    fid_batch_size = args.fid_batch_size

    scores_ret = OrderedDict()

    samples = torch.zeros((evaluation_size, 3, crop_size, crop_size), device=device)
    n_fid_batches = evaluation_size // fid_batch_size

    for i in range(n_fid_batches):
        frm = i * fid_batch_size
        to = frm + fid_batch_size

        if 'DCGAN' in net_type:
            z = torch.randn(fid_batch_size, input_dim, 1, 1, device=device)
        elif 'EGAN' in net_type:
            z = torch.rand(fid_batch_size, input_dim, 1, 1, device=device) * 2. - 1.
        elif net_type == 'WGAN':
            z = torch.randn(fid_batch_size, input_dim, device=device)
        else:
            raise NotImplementedError('netG [%s] is not found' % net_type)

        gen_s = net(z).tanh().detach()
        samples[frm:to] = gen_s
        print("\rgenerate fid sample batch %d/%d " % (i + 1, n_fid_batches), end="", flush=True)

    print("%d samples generating done" % evaluation_size)

    if use_pytorch:
        IS_mean, IS_var, FID = get_inception_metrics(samples, evaluation_size, num_splits=10)

        if test_fid:
            scores_ret['FID'] = float(FID)
        if test_is:
            scores_ret['IS_mean'] = float(IS_mean)
            scores_ret['IS_var'] = float(IS_var)

    else:
        samples = samples.cpu().numpy()
        samples = ((samples + 1.0) * 127.5).astype('uint8')
        samples = samples.reshape(evaluation_size, input_nc, crop_size, crop_size)
        samples = samples.transpose(0, 2, 3, 1)

        if test_fid:
            mu_gen, sigma_gen = fid.calculate_activation_statistics(samples, sess, batch_size=fid_batch_size, verbose=True)
            print("calculate FID:")
            try:
                FID = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
            except Exception as e:
                print(e)
                FID = 500
            scores_ret['FID'] = float(FID)
        if test_is:
            Imlist = []
            for i in range(len(samples)):
                im = samples[i, :, :, :]
                Imlist.append(im)
            print(np.array(Imlist).shape)
            IS_mean, IS_var = get_inception_score(Imlist)
            scores_ret['IS_mean'] = float(IS_mean)
            scores_ret['IS_var'] = float(IS_var)

    return scores_ret
