"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


# def get_params(args, size):
#     w, h = size
#     new_h = h
#     new_w = w
#     if args.preprocess == 'resize_and_crop':
#         new_h = new_w = args.load_size
#     elif args.preprocess == 'scale_width_and_crop':
#         new_w = args.load_size
#         new_h = args.load_size * h // w
#
#     x = random.randint(0, np.maximum(0, new_w - args.crop_size))
#     y = random.randint(0, np.maximum(0, new_h - args.crop_size))
#
#     flip = random.random() > 0.5
#
#     return {'crop_pos': (x, y), 'flip': flip}


def get_transform(args, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in args.preprocess:
        osize = [args.load_size, args.load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in args.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, args.load_size, method)))

    if 'crop' in args.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(args.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], args.crop_size)))

    if args.preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not args.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if convert:
        transform_list += [transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),  # Irrespective of grayscale
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_width:
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if ow > tw or oh > th:
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True


class TorchvisionDataset(data.Dataset):
    """A template dataset class for you to implement custom datasets."""

    def __init__(self, args, train=True):
        """Initialize this dataset class.

        Parameters:
            args (Option class) -- stores all the experiment flags;

        A few things can be done here.
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        self.transform = get_transform(args)

        # import torchvision dataset
        if args.dataset_name == 'CIFAR10':
            from torchvision.datasets import CIFAR10 as torchvisionlib
            self.dataload = torchvisionlib(root=args.download_root, train=train, transform=self.transform,
                                           download=True)
        elif args.dataset_name == 'CIFAR100':
            from torchvision.datasets import CIFAR100 as torchvisionlib
            self.dataload = torchvisionlib(root=args.download_root, train=train, transform=self.transform,
                                           download=True)
        elif args.dataset_name == 'CelebA':
            # from torchvision.datasets import CelebA as torchvisionlib
            # self.dataload = torchvisionlib(root=args.download_root, split='train', transform=self.transform,
            #                                download=True)
            train_transform = transforms.Compose([
                transforms.RandomCrop(178),
                transforms.Resize(args.crop_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5),  # Irrespective of grayscale
                                     (0.5, 0.5, 0.5))
            ])
            from torchvision.datasets import ImageFolder
            self.dataload = ImageFolder(args.download_root+'/img_align_celeba', transform=train_transform)
        elif args.dataset_name == 'LSUN':
            from torchvision.datasets import LSUN as torchvisionlib
            self.dataload = torchvisionlib(root=args.download_root, classes='bedroom_train', transform=self.transform)

        elif args.dataset_name == 'MNIST':
            from torchvision.datasets import MNIST as torchvisionlib
            self.dataload = torchvisionlib(root=args.download_root, train=train, transform=transforms.Compose([
                               transforms.Resize(args.load_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                           ]), download=True)

        else:
            raise ValueError('torchvision_dataset import fault.')

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        """
        item = self.dataload.__getitem__(index)
        return item
        # img = item[0]
        # label = item[1]
        #
        # return {'image': img, 'target': label}

    def __len__(self):
        """Return the total number of images."""
        return len(self.dataload)
