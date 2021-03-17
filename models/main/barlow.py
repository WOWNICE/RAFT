import numpy as np
import torch
import torch.nn as nn

# mapping from the model name to the model constructor
from models import *

import warnings


class Model(nn.Module):
    """
    Reimplementation of the whitening method.
    """
    def __init__(
            self,
            encoder='resnet50',             # models
            projector='2layermlp',
            normalization='l2',
            input_shape=(3, 224, 224),      # shapes
            proj_shape=(4096, 256),
            lmbda=5e-3,
            log=False,                      # log
            **kargs
    ):
        super(Model, self).__init__()

        # load the function from the name
        encoder, projector, normalization = ENCODERS[encoder], HEADS[projector], FUNCS[normalization]

        #################################
        # online network
        online = encoder()
        # deduce the input shape of the projector
        x = torch.from_numpy(np.zeros(shape=[2, *input_shape])).float()
        rep = online(x)
        rep_shape = rep.shape[-1]
        online_proj = projector([rep_shape, *proj_shape])

        self.online = online
        self.online_proj = online_proj

        self.rep_dim = rep_shape

        self.normalize = normalization
        self.lmbda = lmbda

        self.log = log

        self.reps = {}


    def forward(self, x1, x2):
        # generate two randomly-permutated views
        # combine kornia and torchvision.transforms together to boost the performance
        reps1 = self.gen_rep(x1)
        reps2 = self.gen_rep(x2)

        # generate loss
        loss = self.gen_loss(reps1, reps2)

        # register the representations to the model for future estimation
        if self.log:
            self.register_reps(reps1, view=1)
            self.register_reps(reps2, view=2)

        return loss


    def gen_rep(self, x):
        y_online = self.online(x)
        z_online = self.online_proj(y_online)

        return y_online, z_online


    def gen_loss(self, reps1, reps2):
        """
        Different ways of generating losses, overwrites it if needed.
        :param reps1:
        :param reps2:
        :return:
        """
        y_online1, z1 = reps1
        y_online2, z2 = reps2

        z1 = self.normalize(z1 - z1.mean(0))
        z2 = self.normalize(z2 - z2.mean(0))

        c = z1.T.mm(z2) / z1.shape[0]

        c_diff = (c - torch.eye(c.shape[0]).cuda()).square().mul_(self.lmbda)
        loss = (1/self.lmbda - 1)*torch.trace(c_diff) + c_diff.sum()

        return loss


    @torch.no_grad()
    def register_reps(self, reps, view):
        """
        register the representations to the model
        :param reps1:
        :param reps2:
        :return:
        """
        y_online, z_online = reps

        # for estimate uniformity/alignment/others in the future
        # to save the computational cost, we only preserve up to 64 samples

        batch_size = min(y_online.shape[0], 64)
        self.reps[f'online.proj.{view}'] = z_online[:batch_size,:]
        self.reps[f'online.encoder.{view}'] = y_online[:batch_size,:]


if __name__ == '__main__':
    model = Model()
    print(model)