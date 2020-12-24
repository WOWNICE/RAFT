import numpy as np
import torch
import torch.nn as nn

from kornia import augmentation as augs
from kornia import filters, color

# mapping from the model name to the model constructor
from models import *
from models.main import RandomApply

import scipy.spatial as spatial

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
            whiten='cholesky',              # whitening operator
            w_iter=1,
            w_fs=64,
            eps=0,
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
        self.whiten = whiten
        self.w_iter = w_iter
        self.w_fs = w_fs
        self.eps = eps

        self.log = log

        self.reps = {}

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
        y_online1, z_online1 = reps1
        y_online2, z_online2 = reps2

        # perform whitening procedure.
        bs, fs = z_online1.shape
        loss = 0

        # Sliced Whitening,
        # to prevent singular covariance matrix, w_fs < bs-1
        w_fs = min(bs-1, fs, self.w_fs)

        for _ in range(self.w_iter):
            perm = torch.randperm(fs)
            z1, z2 = z_online1[:, perm][:, :w_fs], z_online2[:, perm][:, :w_fs]

            if bs <= z1.shape[1]:
                warnings.warn('sliced feature size <= dimension, might cause singular covariance matrix.', RuntimeWarning)
            try:
                w_z1 = _whiten(z1, eps=self.eps, whiten_method=self.whiten)
                w_z2 = _whiten(z2, eps=self.eps, whiten_method=self.whiten)
            except:
                raise Exception(f'bs={bs}, w_fs={w_fs}.')

            loss += 0.5 * ((self.normalize(z1) - self.normalize(w_z1)).square().mean() +
                           (self.normalize(z2) - self.normalize(w_z2)).square().mean())

        loss /= self.w_iter

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
        self.reps[f'online.{view}'] = self.normalize(z_online)[:batch_size,:]


@torch.no_grad()
def _whiten(x, eps, whiten_method='cholesky'):
    """

    :param x: input tensor x
    :param eps:
        whiten_method=bn:       eps=[sqrt(var + eps)]
        whiten_method=others:   eps=[(1-eps)*cov + eps*eye]
    :param whiten_method:
        whitening methodm, default=cholesky
    :return:
    """
    assert (whiten_method in ['cholesky', 'bn', 'zca'])
    x = x - x.mean(dim=0, keepdims=True)
    x_cov = torch.mm(x.T, x) / (x.shape[0] - 1)

    # make sure it's full-ranked.
    x_cov = (1 - eps) * x_cov + eps * torch.eye(x.shape[1]).cuda()

    if whiten_method == 'cholesky':
        # cholesky is default
        L = torch.cholesky(x_cov)
        W = torch.inverse(L).T
        return torch.mm(x, W)
    elif whiten_method == 'bn':
        x_var = x.square().mean(dim=0)
        return x / torch.sqrt(x_var + eps)
    else:
        raise NotImplementedError(f"{whiten_method} is not implemented yet.")


if __name__ == '__main__':
    model = Model()
    print(model)