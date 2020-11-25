import numpy as np
import torch
import torch.nn as nn

import numpy as np
import torch
import torch.nn as nn

import collections
import importlib
import os
from datetime import datetime
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
    'miniimagenet': 100,
}

def eval(online, args):
    torch.manual_seed(args.rand_seed)

    mock_x = torch.from_numpy(np.zeros(shape=[4, 3, args.resize_dim, args.resize_dim])).float()
    mock_y = online(mock_x)
    rep_dim = mock_y.size(1)

    lr_model = nn.Linear(rep_dim, dataset_classes[args.dataset]).cuda(args.gpu)
    online = online.cuda(args.gpu)
    online.eval() #disable the bn&dropout

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.Adam(lr_model.parameters(), lr=args.lr)

    # set up the dataset
    load_trainset = getattr(importlib.import_module(f'dataset_wrappers.{args.dataset}'), 'load_eval_trainset')
    trainset = load_trainset()
    load_testset = getattr(importlib.import_module(f'dataset_wrappers.{args.dataset}'), 'load_testset')
    testset = load_testset()

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    writer = SummaryWriter()
    last_epoch_accs = []

    for epoch in tqdm(range(args.epochs)):
        metrics = {}
        # train the linear model
        total_samples = 0
        correct_samples = 0
        for step, (images, labels) in enumerate(train_loader):

            # there is additional data augmentation scheme for the
            images, labels = images.cuda(args.gpu), labels.cuda(args.gpu)

            with torch.no_grad():
                reps = online(images)

            logits = lr_model(reps)

            # correctly predicted sample number
            correct_samples += (logits.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)

            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        metrics['train-accuracy'] = correct_samples/total_samples

        total_samples = 0
        correct_samples = 0

        for step, (images, labels) in enumerate(test_loader):
            images, labels = images.cuda(args.gpu), labels.cuda(args.gpu)

            with torch.no_grad():
                reps = online(images)

                logits = lr_model(reps)

                # correctly predicted sample number
                correct_samples += (logits.argmax(1) == labels).sum().item()
                total_samples += labels.size(0)

        metrics['test-accuracy'] = correct_samples / total_samples

        last_epoch_accs.append(correct_samples / total_samples)
        if len(last_epoch_accs) > 10:
            last_epoch_accs = last_epoch_accs[1:]

        # write it into
        writer.add_scalars('accuracy', metrics, epoch)

    print(f'Accuracy of model {args.checkpoint}: {np.array(last_epoch_accs).mean()}±{np.array(last_epoch_accs).std()}')


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10', type=str, metavar='N',
                        help='which dataset to evaluate')
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

    model = load_model(
        model_checkpoint=args.checkpoint,
        encoder=args.encoder,
        online=args.eval_mode == 'online',
        projector=args.projector,
        predictor=args.predictor
    )

    # eval args
    eval(model, args)


