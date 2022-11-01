import numpy as np
import torch
import torch.utils.data
from scipy.special import comb
from envs_repo.torchvision_dataset import TorchvisionDataset
from envs_repo.toy_dataset import Gaussians8Dataset, Gaussians25Dataset, SwissrollDataset


class EnvConstructor:
    """Wrapper around the Environment to expose a cleaner interface for GAN

        This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'ppegan_trainer.py'
    """

    def __init__(self, args):
        """
        A general Environment Constructor
        """
        train_data_loader = TrainDatasetDataLoader(args)
        self.train_dataset = train_data_loader.load_data()


class TrainDatasetDataLoader:
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, args):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.args = args

        self.bs = args.batch_size * args.D_iters
        if 'rsgan' in args.g_loss_mode:
            self.bs += args.batch_size
        if args.dataset_name == '25gaussians' or args.dataset_name == '8gaussians' or args.dataset_name == 'swissroll':
            self.bs += 512
        if args.eval_criteria == 'E-GAN':
            self.bs += args.eval_size

        if args.dataset_name == '25gaussians':
            self.dataset = Gaussians25Dataset()
        elif args.dataset_name == '8gaussians':
            self.dataset = Gaussians8Dataset()
        elif args.dataset_name == 'swissroll':
            self.dataset = SwissrollDataset()
        else:
            self.dataset = TorchvisionDataset(args, True)
        print("train dataset [%s] was created" % type(self.dataset).__name__)

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.bs,  # Load all data for training D.
            shuffle=True,
            num_workers=int(args.num_threads),
            drop_last=True)

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        self.data_size = len(self.dataset)
        return self.data_size

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            # if (i + 1) * self.bs >= self.__len__():
            #     break
            yield data


class TestDatasetDataLoader:
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, args):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.args = args

        self.evalsize = args.eval_size

        self.dataset = TorchvisionDataset(args, False)
        print("test dataset [%s] was created" % type(self.dataset).__name__)

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.evalsize,  # Load all data for evaluating G once together.
            shuffle=True,
            num_workers=int(args.num_threads),
            drop_last=True)

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        self.data_size = len(self.dataset)
        return self.data_size

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            # if (i + 1) * self.evalsize >= self.__len__():
            #     break
            yield data
