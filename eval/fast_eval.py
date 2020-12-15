import numpy as np
import torch
import torch.nn as nn

import collections
import importlib
import os
import time
import argparse
from tqdm import tqdm

import torch.multiprocessing as mp
import torch.distributed as dist

from collections import defaultdict

# supervision of training
from torch.utils.tensorboard import SummaryWriter

from models import name_model_dic

dataset_classes = {
    'cifar10':      10,
    'cifar100':     100,
    'subimagenet': 100,
    'imagenet': 1000,
}

def eval(online, args):
    torch.manual_seed(args.rand_seed)

    mock_x = torch.from_numpy(np.zeros(shape=[4, 3, args.resize_dim, args.resize_dim])).float()
    mock_y = online(mock_x)
    rep_dim = mock_y.size(1)

    lr_model = nn.Linear(rep_dim, dataset_classes[args.dataset]).cuda()
    online = online.cuda()
    online.eval() #disable the bn&dropout

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(lr_model.parameters(), lr=args.lr)

    writer = SummaryWriter()

    # load all the features into one single tensor
    features, labels = [], []
    block_num = 1 + max([int(x.split('-')[2].split('.')[0]) for x in os.listdir(args.cache_dir)])
    for i in range(1, block_num):
        features.append(torch.load(os.path.join(args.cache_dir, f'train-feature-{i}.pt')))
        labels.append(torch.load(os.path.join(args.cache_dir, f'train-label-{i}.pt')))

    # Train on the features
    for epoch in tqdm(range(args.epochs)):
        metrics = {}

        # train the linear model
        total_samples = 0
        correct_samples = 0

        for train_features, train_labels in zip(features, labels):
            train_features, train_labels = train_features.cuda(), train_labels.cuda()
            logits = lr_model(train_features)

            # correctly predicted sample number
            correct_samples += (logits.argmax(1) == train_labels).sum().item()
            total_samples += train_features.size(0)

            loss = criterion(logits, train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        metrics['train-accuracy'] = correct_samples/total_samples

        # write it into
        writer.add_scalars('accuracy', metrics, epoch)


    # Test on test set.

    # set up the dataset
    load_testset = getattr(importlib.import_module(f'dataset_wrappers.{args.dataset}'), 'load_testset')
    testset = load_testset()

    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    total_samples = 0
    correct_samples = 0

    for step, (images, labels) in enumerate(test_loader):
        images, labels = images.cuda(args.gpu), labels.cuda(args.gpu)

        with torch.no_grad():
            reps = online(images)

            logits = lr_model(reps)

            # correctly predicted sample number
            correct_samples += (logits.argmax(1) == labels).sum().item()
            total_samples += images.size(0)

    test_acc = correct_samples / total_samples

    print(f'Accuracy of model {args.checkpoint}: {100*test_acc:.2f}')


def load_model(model_checkpoint, encoder='resnet18', online=True, projector='', predictor=''):
    # load the evaluate
    dic = torch.load(model_checkpoint, map_location=torch.device('cpu'))

    # load the online/target param in a new state dic
    online_param = collections.OrderedDict()
    target_param = collections.OrderedDict()

    for k, v in dic.items():
        if 'online.' in k:
            online_param[k[7:]] = v
        elif 'target.' in k:
            target_param[k[7:]] = v

    encoder = name_model_dic[encoder]()
    if online:
        encoder.load_state_dict(online_param)
    else:
        encoder.load_state_dict(target_param)

    components = [encoder]
    # TODO: support evaluation on projector & predictor
    # if projector:
    #     model.append(name_model_dic[projector]())

    model = nn.Sequential(*components)
    return model


def make_cache(model, cache_dir, dataset, num_batch_in_block):
    # evaluation mode for the model
    model = model.cuda()
    model.eval()

    # set up the dataset
    load_trainset = getattr(importlib.import_module(f'dataset_wrappers.{dataset}'), 'load_eval_trainset')
    trainset = load_trainset()

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    features, labels = [], []
    for step, (image, label) in enumerate(train_loader):
        # there is additional data augmentation scheme for the
        image, label = image.cuda(args.gpu), label.cuda(args.gpu)

        with torch.no_grad():
            features.append(model(image))
            labels.append(label)

        if (step + 1 ) % num_batch_in_block == 0:
            torch.save(torch.cat(features).detach(),
                       os.path.join(cache_dir, f'train-feature-{(step+1)//num_batch_in_block}.pt'))
            torch.save(torch.cat(labels).detach(),
                       os.path.join(cache_dir, f'train-label-{(step+1)//num_batch_in_block}.pt'))

            features, labels = [], []


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10', type=str, metavar='N',
                        help='which dataset to evaluate')
    parser.add_argument('--cache-dir', default='', type=str, metavar='N',
                        help='which directory the features are stored in.')
    parser.add_argument('--cache-blocksize', default=128, type=int, metavar='N',
                        help='blocksize = cache-blocksize * batch-size')

    # models
    parser.add_argument('--encoder', default='resnet18', type=str, metavar='N',
                        help='the encoder type')
    parser.add_argument('--projector', default='', type=str, metavar='N',
                        help='the projector type, if empty, then evaluate w/o projector')
    parser.add_argument('--predictor', default='', type=str, metavar='N',
                        help='the projector type, if empty, then evaluate w/o predictor')

    parser.add_argument('--gpu', default=0, type=int, metavar='N',
                        help='which gpu to use in evaluation')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to train the linear model')
    parser.add_argument('--batch-size', default=512, type=int, metavar='N',
                        help='batch size per node')
    parser.add_argument('--rand-seed', default=2333, type=int, metavar='N',
                        help='default random seed to initialize the model')
    parser.add_argument('--resize-dim', default=32, type=int, metavar='N',
                        help='resize the image dimension')
    parser.add_argument('--lr', default=3e-4, type=float, metavar='N',
                        help='learning rate')
    parser.add_argument('--checkpoint', default='', type=str, metavar='N',
                        help='checkpoint file path')
    parser.add_argument('--num-workers', default=8, type=int, metavar='N',
                        help='how many sub-processes when loading data')
    parser.add_argument('--eval-mode', default='online', type=str, metavar='N',
                        help='which model to be evaluated.')

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)

    # used for estimating the random baseline.
    if args.eval_mode == 'rand':
        model = name_model_dic[args.encoder]()
    else:
        model = load_model(
            model_checkpoint=args.checkpoint,
            encoder=args.encoder,
            online=args.eval_mode == 'online',
            projector=args.projector,
            predictor=args.predictor
        )

    if not args.cache_dir:
        # if ./cache not exist
        if 'cache' not in list(os.listdir('.')):
            os.mkdir('./cache')
        cache_path = f'./cache/{time.strftime("%b-%d_%H-%M-%S", time.localtime())}'
        os.mkdir(cache_path)
        print(f'Created the feature cache path: {cache_path}')

        make_cache(
            model=model,
            cache_dir=cache_path,
            dataset=args.dataset,
            num_batch_in_block=args.cache_blocksize
        )

        args.cache_dir = cache_path

    # eval args
    eval(model, args)


