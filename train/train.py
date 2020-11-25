import numpy as np
import torch
import torch.nn as nn

import time
import os
from datetime import datetime
import argparse
from tqdm import tqdm
import importlib

import torch.multiprocessing as mp
import torch.distributed as dist

import torchvision
import torchvision.transforms as transforms

from torch.utils.data.sampler import SubsetRandomSampler

from collections import defaultdict

# supervision of training
from torch.utils.tensorboard import SummaryWriter


def cleanup():
    dist.destroy_process_group()


def train(gpu, args):
    # import the {data loader/Model} by name
    load_trainset = getattr(importlib.import_module(f'dataset_wrappers.{args.dataset}'), 'load_trainset')
    Model = getattr(importlib.import_module(f'models.main.{args.model}'), 'Model')

    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    torch.manual_seed(args.rand_seed)

    torch.cuda.set_device(gpu)

    # set up the dataset
    # dataset = DatasetDistributed(
    #     data_path='./experiments/cifar10/data',
    #     resize_shape=(args.resize_dim, args.resize_dim),
    #     batch_size=args.batch_size,
    #     device=gpu,
    #     world_size=args.world_size,
    #     rank=rank
    # )

    trainset = load_trainset()

    # trainset = DistributedIndicesWrapper(trainset, list(range(5000)))

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset,
        num_replicas=args.world_size,
        rank=rank
    )

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler
    )

    # BYOL model
    model =  Model(
        # models
        encoder=args.encoder,
        projector=args.projector,
        predictor=args.predictor,
        normalization=args.normalization,
        ema=args.ema_mode,
        ema_lr=args.ema_lr,
        # shapes
        input_shape=(3, args.resize_dim, args.resize_dim),
        proj_shape=(args.mlp_middim, args.mlp_outdim),
        predictor_shape=(args.mlp_middim, args.mlp_outdim)
    ).cuda()

    # if gpu == 0:
    #     print(model)

    # starts from the same initialization.
    if args.same_init == 'True':
        for target_param, online_param in [*zip(model.target.parameters(), model.online.parameters()),
                                           *zip(model.target_proj.parameters(), model.online_proj.parameters())]:
            target_param.data = online_param.data

    # online optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # DDP wrapper for the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    if gpu == 0:
        writer = SummaryWriter()

    # solver
    global_step = 0
    for epoch in range(args.epochs):
        current_time = time.time()
        metrics = defaultdict(list)
        for step, ((x1, x2), labels) in enumerate(train_loader):
            # train the online to approximate the target.
            # each round is formed of 100 epoch
            x1, x2 = x1.cuda(gpu), x2.cuda(gpu)

            for k in range(args.k):
                loss = model(x1, x2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update the target network.
            model.module.update_target()

            if gpu == 0:
                _, loss_alignment, _ = model.module.estimate_align()
                _, loss_cross_model = model.module.estimate_cross()
                _, loss_uniform, _ = model.module.estimate_uniform()
                writer.add_scalars(
                    "Losses/train_step",
                    {
                        'loss-total': loss,
                        'loss-alignment': loss_alignment,
                        'loss-cross-model': loss_cross_model,
                        'loss-uniformity': loss_uniform
                    },
                    global_step
                )

                # append all the loss term into the dictionary.
                metrics["loss-total"].append(loss.item())
                metrics["loss-alignment"].append(loss_alignment.item())
                metrics["loss-cross-model"].append(loss_cross_model.item())
                metrics["loss-uniform"].append(loss_uniform.item())

            global_step += 1

        if gpu == 0:
            # output the time training an epoch
            print(f"Epoch No.{epoch} finished. Time used: {time.time()-current_time}")

            # write metrics to TensorBoard
            dic, var_dic = {}, {}
            for k, v in metrics.items():
                dic[k] = np.array(v).mean()
                # var_dic[k] = np.array(v).std()
            writer.add_scalars("Losses/epoch-loss", dic, epoch)
            # writer.add_scalars("Losses/epoch-stddev", var_dic, epoch)


            if epoch % args.checkpoint_epochs == 0:

                if gpu == 0:
                    print(f"Saving model at epoch {epoch}")
                    torch.save(model.module.state_dict(), f"./{args.checkpoint_dir}/{args.rand_seed}/{args.encoder}_{args.projector}_{args.predictor}_{epoch}.pt")

                # let other workers wait until model is finished
                # dist.barrier()


    # save your improved network
    if gpu == 0:
        torch.save(model.module.state_dict(), f"./{args.checkpoint_dir}/{args.rand_seed}/{args.encoder}_{args.projector}_{args.predictor}_{epoch}.pt")

    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # main model & dataset
    parser.add_argument('--dataset', default='cifar10', type=str, metavar='N',
                        help='which dataset to be trained on.')
    parser.add_argument('--model', default='byol', type=str, metavar='N',
                        help='which model to be trained on.')

    # architecture inside the model
    parser.add_argument('--encoder', default='resnet18', type=str, metavar='N',
                        help='encoder.')
    parser.add_argument('--projector', default='byol-proj', type=str, metavar='N',
                        help='projector.')
    parser.add_argument('--predictor', default='byol-proj', type=str, metavar='N',
                        help='predictor.')
    parser.add_argument('--mlp-middim', default=4096, type=int, metavar='N',
                        help='middle dimension of the mlp.')
    parser.add_argument('--mlp-outdim', default=512, type=int, metavar='N',
                        help='output dimension of the mlp.')
    parser.add_argument('--normalization', default='l2', type=str, metavar='N',
                        help='encoder.')


    # sub-process info
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--port', default='8010', type=str, metavar='N',
                        help='port number')

    # training details
    parser.add_argument('--num-workers', default=4, type=int, metavar='N',
                        help='number of data loader workers per process')
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch-size', default=32, type=int, metavar='N',
                        help='batch size per node')
    parser.add_argument('--rand-seed', default=2333, type=int, metavar='N',
                        help='default random seed to initialize the model')
    parser.add_argument('--resize-dim', default=224, type=int, metavar='N',
                        help='resize the image dimension')
    parser.add_argument('--optimizer', default='adam', type=str, metavar='N',
                        help='which optimizer to optimize the online network')
    parser.add_argument('--lr', default=3e-4, type=float, metavar='N',
                        help='learning rate')
    parser.add_argument('--ema-lr', default=3e-4, type=float, metavar='N',
                        help='ema learning rate')
    parser.add_argument('--ema-mode', default='sgd', type=str, metavar='N',
                        help='how to update the target')
    parser.add_argument('--k', default=1, type=int, metavar='N',
                        help='updates per bootstrap')
    parser.add_argument('--checkpoint-epochs', default=10, type=int, metavar='N',
                        help='checkpoint online model per epoch')
    parser.add_argument('--checkpoint-dir', default='checkpoints', type=str, metavar='N',
                        help='checkpoint online model per epoch')

    parser.add_argument('--same-init', default='False', type=str, metavar='N',
                        help='whether starts from the same initialization')

    # RAFT specific setting
    parser.add_argument('--cross-model-loss-weight', default=1., type=float, metavar='N',
                        help='the weight of the cross model loss')
    parser.add_argument('--align-loss-weight', default=1., type=float, metavar='N',
                        help='the weight of the align model loss')
    parser.add_argument('--cross-model-mode', default='base', type=str, metavar='N',
                        help='mode od the cross-model-loss')


    args = parser.parse_args()
    # print(args)
    args.world_size = args.gpus * args.nodes
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = args.port

    print('Spawning the subprocesses...')
    mp.spawn(train, nprocs=args.gpus, args=(args,), join=True)
