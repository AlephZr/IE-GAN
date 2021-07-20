import torch.utils.data as data
import numpy as np
import sklearn.datasets
import random


class Gaussians25Dataset(data.Dataset):
    """A template dataset class for you to implement custom datasets."""

    def __init__(self):
        """Initialize this dataset class.
        """
        N_POINTS = 5
        RANGE = 2
        centers = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
        centers[:, :, 0] = np.linspace(-RANGE, RANGE, N_POINTS)[:, None]
        centers[:, :, 1] = np.linspace(-RANGE, RANGE, N_POINTS)[None, :]
        centers = centers.reshape((-1, 2))

        scale = 2.
        self.centers = [(scale * x, scale * y) for x, y in centers]

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        """
        point = np.random.randn(2) * 0.05
        center = random.choice(self.centers)
        point[0] += center[0]
        point[1] += center[1]
        point = point.astype('float32')
        point /= 2.828  # stdev

        return point

    def __len__(self):
        """Return the total number of images."""
        return 10000


class Gaussians8Dataset(data.Dataset):
    """A template dataset class for you to implement custom datasets."""

    def __init__(self):
        """Initialize this dataset class.
        """
        centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1. / np.sqrt(2), 1. / np.sqrt(2)),
            (1. / np.sqrt(2), -1. / np.sqrt(2)),
            (-1. / np.sqrt(2), 1. / np.sqrt(2)),
            (-1. / np.sqrt(2), -1. / np.sqrt(2))
        ]

        scale = 2.
        self.centers = [(scale * x, scale * y) for x, y in centers]

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        """
        point = np.random.randn(2) * .02
        center = random.choice(self.centers)
        point[0] += center[0]
        point[1] += center[1]
        point = point.astype('float32')
        point /= 1.414  # stdev

        return point

    def __len__(self):
        """Return the total number of images."""
        return 10000


class SwissrollDataset(data.Dataset):
    """A template dataset class for you to implement custom datasets."""

    def __init__(self):
        """Initialize this dataset class.
        """

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        """
        data = sklearn.datasets.make_swiss_roll(
            n_samples=1,
            noise=0.25
        )[0]
        data = data.astype('float32')[:, [0, 2]][0]
        data /= 7.5  # stdev plus a little

        return data

    def __len__(self):
        """Return the total number of images."""
        return 10000
