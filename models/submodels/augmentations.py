import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import random
from PIL import ImageFilter, Image

from models import AUGS

@AUGS.register_module('small')
class TransformSmallImage:
    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, size):
        s = 1
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        self.train_transform = torchvision.transforms.Compose(
            [
                transforms.RandomResizedCrop(size=size),
                transforms.RandomHorizontalFlip(),  # with 0.5 probability
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize
            ]
        )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)


@AUGS.register_module('mocov1')
class TransformMocov1:
    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, size):
        self.train_transform = torchvision.transforms.Compose(
            [
                transforms.RandomResizedCrop(size, scale=(0.2, 1.)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]
        )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)


@AUGS.register_module('mocov2')
class TransformMocov2:
    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, size):
        self.train_transform = torchvision.transforms.Compose(
            [
                transforms.RandomResizedCrop(size, scale=(0.2, 1.)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                normalize
            ]
        )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)


@AUGS.register_module('byol')
class TransformByol:
    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, size):
        self.view1 = torchvision.transforms.Compose(
            [
                transforms.RandomResizedCrop(size=size),
                transforms.RandomHorizontalFlip(),  # with 0.5 probability
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
                ], p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.),
                torchvision.transforms.ToTensor(),
                normalize,
            ]
        )

        self.view2 = torchvision.transforms.Compose(
            [
                transforms.RandomResizedCrop(size=size),
                transforms.RandomHorizontalFlip(),  # with 0.5 probability
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=.1),
                transforms.RandomApply([Solarization()], p=.2),
                torchvision.transforms.ToTensor(),
                normalize,
            ]
        )

    def __call__(self, x):
        return self.view1(x), self.view2(x)


#utils

normalize = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=(.1, 2.)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarization(object):
    """Solarization augmentation in BYOL https://arxiv.org/abs/2006.07733."""

    def __init__(self, threshold=128):
        self.threshold = threshold

    def __call__(self, img):
        img = np.array(img)
        img = np.where(img < self.threshold, img, 255 -img)
        return Image.fromarray(img.astype(np.uint8))

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str