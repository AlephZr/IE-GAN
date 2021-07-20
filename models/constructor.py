import torch
from core.utils import get_norm_layer, init_net
from core.utils import weights_init, FC_weights_init


class ModelConstructor:

    def __init__(self, args, generator_seed=None, discriminator_seed=None):
        """
        A general Environment Constructor
        """
        self.netD = args.netD
        self.netG = args.netG
        self.nz = args.z_dim
        self.ngf = args.ngf
        self.ndf = args.ndf
        self.nc = args.input_nc
        self.device = args.device
        self.generator_seed = generator_seed
        self.discriminator_seed = discriminator_seed

    def make_model(self, model_type, seed=False):
        """
        Generate and return an model object
        """
        model = None
        # norm_layer = get_norm_layer(norm_type=norm)

        if model_type == 'Discriminator':
            if self.netD == 'DCGAN':
                from models.models import DCGANDiscriminator
                model = DCGANDiscriminator(self.ndf, self.nc).to(device=self.device)
                model.apply(weights_init)
            elif self.netD == 'DCGAN64':
                from models.models import DCGANDiscriminator_64
                model = DCGANDiscriminator_64(self.ndf, self.nc).to(device=self.device)
                model.apply(weights_init)
            elif self.netD == 'DCGAN128':
                from models.models import DCGANDiscriminator_128
                model = DCGANDiscriminator_128(self.ndf, self.nc).to(device=self.device)
                model.apply(weights_init)
            elif self.netG == 'DCGAN28':
                from models.models import DCGANDiscriminator_28
                model = DCGANDiscriminator_28(self.ndf, self.nc).to(device=self.device)
            elif self.netD == 'EGAN32':
                from models.models import EGANDiscriminator_32
                model = EGANDiscriminator_32(ndf=self.ndf, input_nc=self.nc, norm_layer=get_norm_layer('batch'))
                model = init_net(model, device=self.device)
            elif self.netD == 'EGAN64':
                from models.models import EGANDiscriminator_64
                model = EGANDiscriminator_64(ndf=self.ndf, input_nc=self.nc, norm_layer=get_norm_layer('batch'))
                model = init_net(model, device=self.device)
            elif self.netD == 'EGAN128':
                from models.models import EGANDiscriminator_128
                model = EGANDiscriminator_128(ndf=self.ndf, input_nc=self.nc, norm_layer=get_norm_layer('batch'))
                model = init_net(model, device=self.device)
            elif self.netD == 'WGAN':
                from models.models import WGANDiscriminator_cifar10
                model = WGANDiscriminator_cifar10(self.nc, self.ndf).to(device=self.device)
            elif self.netD == 'FC2':
                from models.models import FC2Discriminator
                model = FC2Discriminator(self.ndf).to(device=self.device)
                model.apply(FC_weights_init)
            else:
                raise NotImplementedError('netD [%s] is not found' % self.netD)
            if seed:
                model = torch.load(self.discriminator_seed)
                print('Discriminator seeded from', self.discriminator_seed)

        elif model_type == 'Generator':
            if self.netG == 'DCGAN':
                from models.models import DCGANGenerator
                model = DCGANGenerator(self.nz, self.ngf, self.nc).to(device=self.device)
                model.apply(weights_init)
            elif self.netG == 'DCGAN64':
                from models.models import DCGANGenerator_64
                model = DCGANGenerator_64(self.nz, self.ngf, self.nc).to(device=self.device)
                model.apply(weights_init)
            elif self.netG == 'DCGAN128':
                from models.models import DCGANGenerator_128
                model = DCGANGenerator_128(self.nz, self.ngf, self.nc).to(device=self.device)
                model.apply(weights_init)
            elif self.netG == 'DCGAN28':
                from models.models import DCGANGenerator_28
                model = DCGANGenerator_28(self.nz, self.ngf, self.nc).to(device=self.device)
            elif self.netG == 'EGAN32':
                from models.models import EGANGenerator_32
                model = EGANGenerator_32(z_dim=self.nz, ngf=self.ngf, output_nc=self.nc,
                                         norm_layer=get_norm_layer('none'))
                model = init_net(model, device=self.device)
            elif self.netG == 'EGAN64':
                from models.models import EGANGenerator_64
                model = EGANGenerator_64(z_dim=self.nz, ngf=self.ngf, output_nc=self.nc,
                                         norm_layer=get_norm_layer('none'))
                model = init_net(model, device=self.device)
            elif self.netG == 'EGAN128':
                from models.models import EGANGenerator_128
                model = EGANGenerator_128(z_dim=self.nz, ngf=self.ngf, output_nc=self.nc,
                                          norm_layer=get_norm_layer('none'))
                model = init_net(model, device=self.device)
            elif self.netG == 'WGAN':
                from models.models import WGANGenerator_cifar10
                model = WGANGenerator_cifar10(self.nz, self.nc, self.ngf).to(device=self.device)
            elif self.netG == 'FC2':
                from models.models import FC2Generator
                model = FC2Generator(self.nz, self.ngf).to(device=self.device)
                model.apply(FC_weights_init)
            else:
                raise NotImplementedError('netG [%s] is not found' % self.netG)
            if seed:
                model = torch.load(self.generator_seed)
                print('Generator seeded from', self.generator_seed)

        else:
            AssertionError('Unknown model type')

        return model
