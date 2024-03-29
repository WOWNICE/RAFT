import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms


def load_trainset(trans):
    """
    Training set during SSL
    :return:
    """
    return torchvision.datasets.CIFAR10(
        root='./data/cifar10',
        train=True,
        download=False,
        transform=trans(32)
    )

def load_eval_trainset(**kwargs):
    """
    Training set during Linear Evaluation Protocol
    :return:
    """
    return torchvision.datasets.CIFAR10(
        root='./data/cifar10',
        train=True,
        download=False,
        transform=torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(size=32),
                torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
    )

def load_testset():
    """
    Test set during the
    :return:
    """
    return torchvision.datasets.CIFAR10(
        root='./data/cifar10',
        train=False,
        download=False,
        transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
    )


if __name__ == '__main__':
    trainset = load_trainset()
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    train_iter = iter(train_loader)
    x = train_iter.next()
    print(x)