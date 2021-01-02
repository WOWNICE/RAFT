# Implementation of the Bipartite Graph Matching method
import numpy as np
import torch
import torch.nn as nn

# mapping from the model name to the model constructor
from models import *

import warnings

try:
    import lapsolverc as lapsolver
except:
    import lapsolver


class Model(nn.Module):
    """
    Implementation of the optimal Bipartite Graph Matching for Self-Supervised Learning.
    """
    def __init__(
            self,
            encoder='resnet50',             # models
            projector='2layermlp',
            normalization='l2',
            input_shape=(3, 224, 224),      # shapes
            proj_shape=(4096, 256),
            solver='lap',                   # bgm-solver
            prior='gaussian',               # prior distribution
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
        self.solver = solver
        self.prior = prior

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

        # sample a batch of targets from the prior
        sample1, sample2 = self.sample_prior(z1.shape), self.sample_prior(z2.shape)

        # now move to cpu to solve the lap problem
        z1, z2 = z1 - z1.mean(dim=0, keepdims=True), z2 - z2.mean(dim=0, keepdims=True)
        target1, target2 = self.match_prior(z1, sample1, solver=self.solver), self.match_prior(z2, sample2, solver=self.solver)

        loss = 0.5 * ((self.normalize(z1) - self.normalize(target2)).square().sum(dim=1).mean() +
                       (self.normalize(z2) - self.normalize(target1)).square().sum(dim=1).mean())

        return loss

    def match_prior(self, x, s, solver):
        # build cost matrix
        # 1. -x^T s
        # 2. cij = ||xi - sj||
        c = (-torch.mm(x, s.T)).detach().cpu().numpy()

        if solver == 'dense':
            _, cids = lapsolver.solve_dense(c)
        else:
            raise NotImplementedError(f'{solver} not implemented.')

        return s[cids]

    def sample_prior(self, size):
        assert self.prior in prior_dict
        return prior_dict[self.prior](size)

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


#######################################
# utils
# prior sampling
@torch.no_grad()
def uniform_prior(size):
    """
    return a uniform distribution of [-1, 1]
    :param shape:
    :return:
    """
    return 2*(torch.rand(size=size) - 0.5).cuda()

@torch.no_grad()
def gaussian_prior(size):
    """
    return a gaussian distribution of mean=0, std=1.
    :param size:
    :return:
    """
    return torch.normal(mean=0, std=1, size=size).cuda()

prior_dict = {
    'uniform': uniform_prior,
    'gaussian': gaussian_prior
}


if __name__ == '__main__':
    model = Model()
    print(model)