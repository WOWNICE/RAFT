import numpy as np
import torch
import torch.nn as nn
import argparse
from torch.multiprocessing import Process, Queue
import importlib
import time
from .utils import *

def knn(x_train, y_train, x_test, y_test, k=5,):
    """ k-nearest neighbors classifier accuracy """

    d = torch.cdist(x_test, x_train)
    topk = torch.topk(d, k=k, dim=1, largest=False)
    labels = y_train[topk.indices]
    pred = torch.empty_like(y_test)
    for i in range(len(labels)):
        x = labels[i].unique(return_counts=True)
        pred[i] = x[0][x[1].argmax()]

    acc = (pred == y_test).float().mean().cpu().item()
    del d, topk, labels, pred
    return acc


def eval_knn(model, args):
    # prep
    load_trainset = getattr(importlib.import_module(f'dataset_apis.{args.dataset}'), 'load_eval_trainset')
    trainset = load_trainset()
    inds = np.array_split(range(len(trainset)), args.gpus)
    train_loaders = [torch.utils.data.DataLoader(
        torch.utils.data.Subset(trainset, ind),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    ) for ind in inds]

    load_testset = getattr(importlib.import_module(f'dataset_apis.{args.dataset}'), 'load_testset')
    testset = load_testset()
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # multi-processing with gpus
    # pool = Pool()
    # func_args = [(gpu, model, train_loader, test_loader, args.k)
    #     for (gpu, train_loader) in enumerate(train_loaders)
    # ]
    # print(func_args)
    # lst = pool.starmap(load_tensor_single, func_args)
    queue, exit_queue = Queue(), Queue()
    processes = [
        Process(target=load_tensor_single,
                args=(gpu, model, train_loader, test_loader, args.k, queue, exit_queue))
        for (gpu, train_loader) in enumerate(train_loaders)
    ]

    ##########################################
    for p in processes:
        p.start()

    # synchronization
    lst = []
    for i in range(len(processes)):
        item = queue.get()
        lst.append(item)
        exit_queue.put(item[0]) # put the pid back.

    for p in processes:
        p.join()
    ##########################################

    labels = torch.cat([item[1].cuda(0) for item in lst], dim=1)
    distances = torch.cat([item[2].cuda(0) for item in lst], dim=1)
    test_y = lst[0][-1].cuda(0)

    # print(labels.shape, distances.shape, test_y.shape)

    topk = torch.topk(distances, dim=1, k=args.k, largest=False)
    labels = torch.cat([labels[range(labels.shape[0]), topk.indices[:,i]].expand(1,-1).T for i in range(topk.indices.shape[1])], dim=1)
    pred = torch.empty_like(test_y)
    for i in range(len(labels)):
        x = labels[i].unique(return_counts=True)
        pred[i] = x[0][x[1].argmax()]

    acc = (pred == test_y).float().mean().item()
    print(f"[TEST]\t[KNN-ACC={acc*100.:3.2f}%]")


@torch.no_grad()
def load_tensor_single(gpu, model, train_loader, test_loader, k, queue, exit_queue):
    model = model.cuda(gpu)

    # training features
    xs, ys = [], []
    for step, (images, labels) in enumerate(train_loader):
        images, labels = images.cuda(gpu, non_blocking=True), labels.cuda(gpu, non_blocking=True)

        reps = model(images)

        xs.append(reps)
        ys.append(labels)

    train_x, train_y = torch.cat(xs), torch.cat(ys)
    del xs, ys

    # test features
    xs, ys = [], []
    for step, (images, labels) in enumerate(test_loader):
        images, labels = images.cuda(gpu, non_blocking=True), labels.cuda(gpu, non_blocking=True)

        reps = model(images)

        xs.append(reps)
        ys.append(labels)

    test_x, test_y = torch.cat(xs), torch.cat(ys)
    del xs, ys

    # compute local knn
    d = torch.cdist(test_x, train_x)
    topk = torch.topk(d, k=k, dim=1, largest=False)
    labels = train_y[topk.indices]
    distances = topk.values

    # put to the queue, test_y is global
    # there is some issue with queue.get(torch.Tensor)
    queue.put((gpu, labels.cpu(), distances.cpu(), test_y.cpu()))

    # manual synchronization
    while True:
        allow_exit = exit_queue.get()
        if allow_exit != gpu:
            # print(f'proc.{gpu} get key.{allow_exit}, put it back.')
            exit_queue.put(allow_exit)
            time.sleep(1)
        else:
            # print(f'proc.{gpu} get key.{allow_exit}, exiting.')
            break

    return




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
    parser.add_argument('--batch-size', default=512, type=int, metavar='N',
                        help='batch size per node')
    parser.add_argument('--resize-dim', default=32, type=int, metavar='N',
                        help='resize the image dimension')
    parser.add_argument('--checkpoint', default='', type=str, metavar='N',
                        help='checkpoint file path')
    parser.add_argument('--num-workers', default=8, type=int, metavar='N',
                        help='how many sub-processes when loading data')
    parser.add_argument('--eval-mode', default='online', type=str, metavar='N',
                        help='which model to be evaluated.')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='split the training features into {gpus} parts.')
    parser.add_argument('--k', default=5, type=int,
                        help='k neighbors.')

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
            online=args.eval_mode == 'online',
            projector=args.projector,
            predictor=args.predictor
        )

    eval_knn(model=model, args=args)
