import numpy as np
import torch
import torch.nn as nn

# mapping from the model name to the model constructor
from models import *
from models.main import RandomApply

import scipy.spatial as spatial


class Model(nn.Module):
    """
    BYOL model.
    """
    def __init__(
            self,
            # models
            encoder='resnet18',
            projector='2layermlpbn',
            predictor='2layermlpbn',
            normalization='l2',

            # shapes
            input_shape=(3, 224, 224),
            proj_shape=(4096, 256),
            predictor_shape=(4096, 256),
            same_init=False,
            log=False,
            **kargs
    ):
        super(Model, self).__init__()

        # load the function from the name
        encoder, projector, predictor, normalization = ENCODERS[encoder], HEADS[projector], HEADS[predictor], FUNCS[normalization]

        #################################
        # target network
        target = encoder()
        # deduce the input shape of the projector
        x = torch.from_numpy(np.zeros(shape=[2, *input_shape])).float()
        rep = target(x)
        rep_shape = rep.shape[-1]
        target_proj = projector([rep_shape, *proj_shape])

        # stop gradient
        for target_param in [*target.parameters(), *target_proj.parameters()]:
            target_param.requires_grad = False

        self.target = target
        self.target_proj = target_proj

        ##################################
        # create the online network
        self.online = encoder()
        self.online_proj = projector([rep_shape, *proj_shape])

        # online-target initialization
        if same_init:
            for target_param, online_param in [*zip(self.target.parameters(), self.online.parameters()),
                                               *zip(self.target_proj.parameters(), self.online_proj.parameters())]:
                target_param.data = online_param.data

        self.pred = predictor([proj_shape[-1], *predictor_shape])

        self.rep_dim = rep_shape

        self.normalize = normalization

        # whether
        self.log = log

        self.reps = {}

    def gen_rep(self, x):
        y_online = self.online(x)
        z_online = self.online_proj(y_online)
        z_online_pred = self.pred(z_online)

        with torch.no_grad():
            y_target = self.target(x)
            z_target = self.target_proj(y_target)

        return y_online, z_online, z_online_pred, y_target, z_target

    def gen_loss(self, reps1, reps2):
        """
        Different ways of generating losses, overwrites it if needed.
        :param reps1:
        :param reps2:
        :return:
        """
        y_online1, z_online1, z_online_pred1, y_target1, z_target1 = reps1
        y_online2, z_online2, z_online_pred2, y_target2, z_target2 = reps2

        # normalize them
        z_online_pred1 = self.normalize(z_online_pred1)
        z_online_pred2 = self.normalize(z_online_pred2)
        z_target1 = self.normalize(z_target1)
        z_target2 = self.normalize(z_target2)

        # compute the loss
        loss = ((z_online_pred1 - z_target2).square().mean() + (z_online_pred2 - z_target1).square().mean()) / 2

        return loss

    @torch.no_grad()
    def register_reps(self, reps, num=0):
        """
        register the representations to the model
        :param reps1:
        :param reps2:
        :return:
        """
        y_online, z_online, z_online_pred, y_target, z_target = reps

        # for estimate uniformity/alignment/others in the future
        # to save the computational cost, we only preserve up to 64 samples

        batch_size = min(y_online.shape[0], 64)

        self.reps[f'online.{num}'] = self.normalize(z_online)[:batch_size,:]
        self.reps[f'online.pred.{num}'] = self.normalize(z_online_pred)[:batch_size,:]
        self.reps[f'target.{num}'] = self.normalize(z_target)[:batch_size,:]

    def forward(self, x1, x2):
        """
        loss = loss_consistency + loss_cross_model + loss_cross_term
        :param x:
        :return: three loss
        """
        # generate two randomly-permutated views
        # combine kornia and torchvision.transforms together to boost the performance
        reps1 = self.gen_rep(x1)
        reps2 = self.gen_rep(x2)

        # generate loss
        loss = self.gen_loss(reps1, reps2)

        # register the representations to the model for future estimation
        if self.log:
            self.register_reps(reps1, num=1)
            self.register_reps(reps2, num=2)

        return lossi


if __name__ == '__main__':
    model = Model()
    print(model)