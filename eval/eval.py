import numpy as np
import torch
import torch.nn as nn

import collections
import importlib
import os
import argparse
from tqdm import tqdm

# supervision of training
from torch.utils.tensorboard import SummaryWriter

import torch.multiprocessing as mp
import torch.distributed as dist

from torch.utils.data.sampler import SubsetRandomSampler

# supervision of training
from torch.utils.tensorboard import SummaryWriter

from .utils import *

def cleanup():
    dist.destroy_process_group()


def eval(gpu, online, args):
    # dist stuff
    if args.distributed:
        rank = args.nr * args.gpus + gpu
        dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    torch.manual_seed(args.rand_seed)

    torch.cuda.set_device(gpu)

    # mocking input and get rep-dim dynamically.
    mock_x = torch.from_numpy(np.zeros(shape=[4, 3, args.resize_dim, args.resize_dim])).float()
    mock_y = online(mock_x)
    rep_dim = mock_y.size(1)

    lr_model = nn.Linear(rep_dim, dataset_classes[args.dataset]).cuda()
    lr_model.bias.data.zero_()
    online = online.cuda()
    online.eval() #disable the bn&dropout

    # DDP wrapper for the model
    # online = nn.parallel.DistributedDataParallel(online, device_ids=[gpu])
    if args.distributed:
        lr_model = nn.parallel.DistributedDataParallel(lr_model, device_ids=[gpu])

    # criterion and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(lr_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(lr_model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    # lr scheduling
    # compute batch size / global batch size and other helpful information
    global_bs = args.batch_size * args.gpus

    total_steps = args.epochs * total_samples_dict[args.dataset] // global_bs
    warmup_steps = args.warmup_epochs * total_samples_dict[args.dataset] // global_bs

    optimizer = WRAPPERS[args.lr_wrapper](optimizer, total_steps=total_steps, warmup_steps=warmup_steps, c_step=0)

    #############################
    # TRAINING
    # set up the dataset
    load_trainset = getattr(importlib.import_module(f'dataset_apis.{args.dataset}'), 'load_eval_trainset')
    trainset = load_trainset()
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset,
        num_replicas=args.world_size,
        rank=rank,
        shuffle=True,
    ) if args.distributed else None

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler
    )

    if gpu == 0:
        writer = SummaryWriter()
        print(f"[LOG]\t{writer.log_dir}")

    metrics = {}
    for epoch in tqdm(range(args.epochs)):
        # train the linear model
        correct_top1, correct_top5, total_samples = 0, 0, 0
        for step, (images, labels) in enumerate(train_loader):

            # there is additional data augmentation scheme for the
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)

            with torch.no_grad():
                reps = online(images)

            logits = lr_model(reps)

            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # correctly predicted sample number
            top1, top5 = correct_k(logits, labels, (1, 5))
            correct_top1 += top1
            correct_top5 += top5
            total_samples += labels.size(0)

        metrics['train-top1'] = 100 * correct_top1 / total_samples
        metrics['train-top5'] = 100 * correct_top5 / total_samples

        if gpu == 0:
            writer.add_scalars('accs(%)', metrics, epoch)

    ##############################
    # TESTING
    load_testset = getattr(importlib.import_module(f'dataset_apis.{args.dataset}'), 'load_testset')
    testset = load_testset()

    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    correct_top1, correct_top5, total_samples = 0, 0, 0
    for step, (images, labels) in enumerate(test_loader):
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)

        with torch.no_grad():
            reps = online(images)

            logits = lr_model(reps)

            # correctly predicted sample number
            top1, top5 = correct_k(logits, labels, topk=(1, 5))
            correct_top1 += top1
            correct_top5 += top5
            total_samples += labels.size(0)

    top1_acc = (correct_top1 / total_samples).cpu().numpy()[0]
    top5_acc = (correct_top5 / total_samples).cpu().numpy()[0]

    if gpu == 0:
        print(f"[TEST]\t[TOP1={top1_acc*100.:3.2f}%]\t[TOP5={top5_acc*100.:3.2f}%]")

    if args.distributed: cleanup()





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

    # training details
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to train the linear model')
    parser.add_argument('--warmup-epochs', default=0, type=int, metavar='N',
                        help='number of total epochs to train the linear model')
    parser.add_argument('--batch-size', default=512, type=int, metavar='N',
                        help='batch size per node')
    parser.add_argument('--rand-seed', default=2333, type=int, metavar='N',
                        help='default random seed to initialize the model')
    parser.add_argument('--resize-dim', default=32, type=int, metavar='N',
                        help='resize the image dimension')
    parser.add_argument('--lr', default=3e-3, type=float, metavar='N',
                        help='learning rate')
    parser.add_argument('--checkpoint', default='', type=str, metavar='N',
                        help='checkpoint file path')
    parser.add_argument('--num-workers', default=8, type=int, metavar='N',
                        help='how many sub-processes when loading data')
    parser.add_argument('--eval-mode', default='online', type=str, metavar='N',
                        help='which model to be evaluated.')
    parser.add_argument('--optimizer', default='sgd', type=str, metavar='N',
                        help='optimizer')
    parser.add_argument('--weight-decay', default=0., type=float, metavar='N',
                        help='weight decay for the network learning')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='N',
                        help='momentum for the sgd optimizer.')
    parser.add_argument('--lr-wrapper', default='empty', type=str, metavar='N',
                        help='momentum for the sgd optimizer.')

    # sub-process info
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--port', default='8010', type=str, metavar='N',
                        help='port number')
    parser.add_argument('--distributed', action='store_true',
                        help='whether use distributed mode')

    args = parser.parse_args()

    # used for estimating the random baseline.
    if args.eval_mode == 'rand':
        print(f"[LOAD]\trandom baseline")
        model = ENCODERS[args.encoder]()
    else:
        print(f"[LOAD]\t{args.checkpoint}")
        model = load_model(
            model_checkpoint=args.checkpoint,
            encoder=args.encoder,
            mode=args.eval_mode,
            projector=args.projector,
            predictor=args.predictor
        )

    if args.distributed:
        args.world_size = args.gpus * args.nodes
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = args.port

        print(f'[SPAWN]\t[PORT={args.port}]')
        mp.spawn(eval, nprocs=args.gpus, args=(model, args,), join=True)

    else:
        # eval args
        # eval(model, args)
        eval(gpu=0, online=model, args=args)



