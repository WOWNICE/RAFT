import numpy as np
import torch
import torch.nn as nn

from kornia import augmentation as augs
from kornia import filters, color

# mapping from the model name to the model constructor
from models import *
from models.main import RandomApply

import scipy.spatial as spatial


class Model(nn.Module):
    """
    Reimplementation of the whitening method.
    """
    def __init__(
            self,
            # models
            encoder='resnet50',
            projector='2layermlp',
            normalization='l2',

            # shapes
            input_shape=(3, 224, 224),
            proj_shape=(4096, 256),

            # whitening operater.
            whiten='cholesky',
            whiten_grad=False,
            w_iter=1,
            w_split=2,
            eps=0,

            # log.
            log=False,
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
        self.whiten_grad = whiten_grad
        self.w_iter = w_iter
        self.w_split = w_split
        self.eps = eps

        self.log = log

        # set the transformations
        self._set_transforms(input_shape[1])

    def gen_rep(self, x):
        view = self.train_augment(x)
        y_online = self.online(view)
        z_online = self.online_proj(y_online)

        return y_online, z_online

    def whitening(self, x):
        x = x - x.mean(dim=0, keepdims=True)
        # supppose x is already 0-mean
        x_cov = torch.mm(x.T, x) / (x.shape[0] - 1)

        # make sure it's full-ranked.
        x_cov = (1-self.eps) * x_cov + self.eps * torch.eye(x.shape[1]).cuda()

        if self.whiten_grad:
            if self.whiten == 'cholesky':
                # cholesky is default
                L = torch.cholesky(x_cov)
                W = torch.inverse(L).T
                return torch.mm(x, W)
            elif self.whiten == 'bn':
                x_var = x.square().mean(dim=0)
                return x / torch.sqrt(x_var + self.eps)
            else:
                raise NotImplementedError(f"{self.whiten} is not implemented yet.")

        # if the gradient of the cholsky not passed to the encoder.
        with torch.no_grad():
            if self.whiten == 'cholesky':
                # cholesky is default
                L = torch.cholesky(x_cov)
                W = torch.inverse(L).T
            elif self.whiten == 'bn':
                x_var = x.square().mean(dim=0)
                return x / torch.sqrt(x_var + self.eps)
            else:
                raise NotImplementedError(f"{self.whiten} is not implemented yet.")

        return torch.mm(x, W)

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
        total_splits = 0
        for _ in range(self.w_iter):
            perm = torch.randperm(bs)
            z1, z2 = z_online1[perm], z_online2[perm]

            # handle possible exceptions when batch_slice_size < 2*dim
            batch_slice_size = bs // self.w_split
            remain = bs % batch_slice_size
            if min(batch_slice_size, remain if remain!=0 else float('inf')) < 1.9 * fs:
                return 0 * torch.norm(self.normalize(z1), p=2)  # not making any effect
            try:
                y1s = torch.split(z1, batch_slice_size)
                z1s = [self.whitening(x) for x in y1s]
                y2s = torch.split(z2, batch_slice_size)
                z2s = [self.whitening(x) for x in y2s]
            except:
                raise Exception(f'bs={bs}, fs={fs}, batch-slice-size={batch_slice_size}, remain={remain}')
            total_splits += len(z1s)

            # adding the stop-gradient
            for i in range(len(y1s)):
                loss += 0.5 * ((self.normalize(y1s[i]) - self.normalize(z2s[i].detach())).square().mean() +
                               (self.normalize(y2s[i]) - self.normalize(z1s[i].detach())).square().mean())

        loss /= (self.w_iter * total_splits)

        return loss

    @torch.no_grad()
    def register_reps(self, reps1, reps2):
        """i
        register the representations to the model
        :param reps1:
        :param reps2:
        :return:
        """
        y_online1, z_online1 = reps1
        y_online2, z_online2 = reps2

        # for estimate uniformity/alignment/others in the future
        # to save the computational cost, we only preserve up to 64 samples

        batch_size = min(y_online2.shape[0], 64)
        self.z_online1 = self.normalize(z_online1)[:batch_size,:]
        self.z_online2 = self.normalize(z_online2)[:batch_size,:]


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
            self.register_reps(reps1, reps2)

        return loss

    @torch.no_grad()
    def estimate_align(self):
        """
        Estimate the online/pred/target's alignment AFTER normalization.
        :return:
        """

        align_online = (self.z_online1 - self.z_online2).square().mean().detach()

        return align_online

    @torch.no_grad()
    def estimate_uniform(self):
        """
        Estimate the online/pred/target uniformity AFTER normalization.
        :return:
        """

        def lunif(x, t=2):
            x = x.cpu().numpy()
            # sq_pdist = torch.pdist(x, p=2).pow(2)     # not supported in AMP
            sq_pdist = torch.Tensor(spatial.distance.pdist(x, 'minkowski', p=2)).pow(2)
            return sq_pdist.mul(-t).exp().mean().log()

        uniform_online = 0.5 * (lunif(self.z_online1) + lunif(self.z_online2))

        return uniform_online

    def _set_transforms(self, resize_dim=224):
        self.train_augment = nn.Sequential(
            # augs.RandomResizedCrop((resize_dim, resize_dim)),           # 1
            # augs.RandomHorizontalFlip(),                                # 2
            # RandomApply(augs.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.8),   # 3
            # augs.RandomGrayscale(p=0.2),                                # 4
            # RandomApply(filters.GaussianBlur2d((3, 3), (1.5, 1.5)), p=0.1),
            # in kornia: should be a tuple
            # color.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))
        )


if __name__ == '__main__':
    model = Model()
    print(model)