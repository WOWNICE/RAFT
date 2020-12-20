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


from models import * #import all the registries


def cleanup():
    dist.destroy_process_group()


def train(gpu, args):
    # import the {data loader/Model} by name
    load_trainset = getattr(importlib.import_module(f'dataset_apis.{args.dataset}'), 'load_trainset')
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
        rank=rank,
        shuffle=True,
    )

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler
    )

    # model
    model =  Model(
        # models
        encoder=args.encoder,
        projector=args.projector,
        predictor=args.predictor,
        normalization=args.normalization,

        # shapes
        input_shape=(3, args.resize_dim, args.resize_dim),
        proj_shape=(args.mlp_middim, args.mlp_outdim),
        predictor_shape=(args.mlp_middim, args.mlp_outdim),
        alignment_weight=args.alignment_weight,
        cross_weight=args.cross_weight,

        # init & log
        same_init=args.same_init == 'True',
        log=args.log=='True',

        # whitening arguments
        eps=args.whiten_eps,
        whiten=args.whiten,
        whiten_grad=args.whiten_grad=='True',
        w_iter = args.w_iter,
        w_split = args.w_split
    )

    # reload the checkpoint
    if args.reload_ckpt:
        dic = torch.load(args.reload_ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(dic)

    # sync batch
    if args.sync_bn == 'True':
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
    else:
        model = model.cuda()

    # online optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # amp apex training for the main model
    try:
        from apex import amp
    except:
        args.amp = False
    if args.amp == 'True':
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    # ema wrapper should go after the amp wrapper
    WeightWrapper = WRAPPERS[args.weight_wrapper]
    model = WeightWrapper(model, opt=args.ema_mode, lr=args.ema_lr, momentum=1, pred_path=args.pred_checkpoint)
    # _, model.optimizer = amp.initialize([model.model.target, model.model.target_proj], model.optimizer, opt_level="O1")

    # a series of logger wrappers
    Loggers = [WRAPPERS[x] for x in args.logger_wrappers.split('.')]
    for Logger in Loggers:
        model = Logger(model)

    # DDP wrapper for the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    if gpu == 0 and args.log == 'True':
        writer = SummaryWriter()
        print(f"[LOG]\t{writer.log_dir}")

    # solver
    global_step = 0
    for epoch in range(args.reload_epoch, args.epochs+args.reload_epoch):
        current_time = time.time()
        model.module.clear()    # clear the metrics dic
        for step, ((x1, x2), labels) in enumerate(train_loader):
            # train the online to approximate the target.
            # each round is formed of 100 epoch
            x1, x2 = x1.cuda(non_blocking=True), x2.cuda(non_blocking=True)


            for k in range(args.k):
                loss = model(x1, x2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update the target network.
            model.module.update()

            if gpu == 0 and args.log == 'True':
                model.module.estimate()

            global_step += 1

        if gpu == 0:
            # output the time training an epoch
            print(f"[TRAIN]\t[EPOCH={epoch:04}]\t[TIME={time.time()-current_time:5.2f}s]")

            # write metrics to TensorBoard
            if args.log == 'True':
                dic, var_dic = {}, {}
                for k, v in model.module.metrics.items():
                    dic[k] = np.array(v).mean()
                    # var_dic[k] = np.array(v).std()
                writer.add_scalars("Losses/epoch", dic, epoch)
                # writer.add_scalars("Losses/epoch-stddev", var_dic, epoch)


            if epoch % args.checkpoint_epochs == 0:

                if gpu == 0:
                    print(f"[CKPT]\t[EPOCH={epoch:04}]\t[PATH={args.checkpoint_dir}/{args.rand_seed}]")
                    # model -> [weight_wrapper, ..., foo_wrapper] -> ddp_wrapper
                    torch.save(model.module.module.state_dict(), f"./{args.checkpoint_dir}/{args.rand_seed}/{args.encoder}_{args.projector}_{args.predictor}_{epoch}.pt")

                # let other workers wait until model is finished
                # dist.barrier()


    # save your improved network
    if gpu == 0:
        # model -> [weight_wrapper, ..., foo_wrapper] -> ddp_wrapper
        torch.save(model.module.module.state_dict(), f"./{args.checkpoint_dir}/{args.rand_seed}/{args.encoder}_{args.projector}_{args.predictor}_{epoch}.pt")

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
    parser.add_argument('--projector', default='2layermlpbn', type=str, metavar='N',
                        help='projector.')
    parser.add_argument('--predictor', default='2layermlpbn', type=str, metavar='N',
                        help='predictor.')
    parser.add_argument('--mlp-middim', default=4096, type=int, metavar='N',
                        help='middle dimension of the mlp.')
    parser.add_argument('--mlp-outdim', default=512, type=int, metavar='N',
                        help='output dimension of the mlp.')
    parser.add_argument('--normalization', default='l2', type=str, metavar='N',
                        help='normalization function')
    parser.add_argument('--weight-wrapper', default='ema', type=str, metavar='N',
                        help='registered in models.WRAPPERS')
    parser.add_argument('--logger-wrappers', default='emptylogger', type=str, metavar='N',
                        help="a list of loggers, separated by '.'")


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
    parser.add_argument('--log', default='False', type=str, metavar='N',
                        help='whether logs the stats')
    parser.add_argument('--sync-bn', default='False', type=str, metavar='N',
                        help='whether globally sync BN layers')
    parser.add_argument('--amp', default='False', type=str, metavar='N',
                        help='whether use apex.amp to accelerate training')
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
    parser.add_argument('--reload-ckpt', default='', type=str, metavar='N',
                        help='which checkpoint to reload.')
    parser.add_argument('--reload-epoch', default=0, type=int, metavar='N',
                        help='epoch trained on the reloaded checkpoint.')

    parser.add_argument('--same-init', default='False', type=str, metavar='N',
                        help='whether starts from the same initialization')

    # RAFT specific setting
    parser.add_argument('--cross-weight', default=1., type=float, metavar='N',
                        help='the weight of the cross model loss')
    parser.add_argument('--alignment-weight', default=1., type=float, metavar='N',
                        help='the weight of the align model loss')
    parser.add_argument('--cross-model-mode', default='base', type=str, metavar='N',
                        help='mode od the cross-model-loss')

    # wrappers additional setting
    parser.add_argument('--pred-checkpoint', default='', type=str, metavar='N',
                        help='which predictor to load')

    # settings for regalign model
    parser.add_argument('--p-norm', default=2, type=int, metavar='N',
                        help='the p_norm constraint on the regalign model.')
    parser.add_argument('--reg-weight', default=1., type=float, metavar='N',
                        help='the loss weight on the predictor regularizer.')

    # setting for the whitening model
    parser.add_argument('--whiten', default='cholesky', type=str, metavar='N',
                        help='which whitening method to use')
    parser.add_argument('--whiten-grad', default='True', type=str, metavar='N',
                        help='whether the gradient of decomposition is passed to the encoder.')
    parser.add_argument('--whiten-eps', default=0., type=float, metavar='N',
                        help='the epsilon value stablizing the decomposition.')
    parser.add_argument('--w-iter', default=1, type=int, metavar='N',
                        help='time of repeating the whitening loss for stablizing the loss.')
    parser.add_argument('--w-split', default=1, type=int, metavar='N',
                        help='split total batch into w_split sub-batches.')

    args = parser.parse_args()
    # print(args)
    args.world_size = args.gpus * args.nodes
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = args.port

    print(f'[SPAWN]\t[PORT={args.port}]')
    mp.spawn(train, nprocs=args.gpus, args=(args,), join=True)
