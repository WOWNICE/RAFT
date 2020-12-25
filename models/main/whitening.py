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
            whiten='cholesky',              # whitening operator
            w_iter=0,                       # 0: adaptive w_iter
            w_fs=0,                         # 0: adaptive w_fs
            queue_size=1024,
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

        # queue for whitening computation
        # credit to MoCo code
        self.queue_size = queue_size
        self.queue_full = False
        self.register_buffer("queue", torch.randn(queue_size, proj_shape[-1]))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

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
        y_online1, z1 = reps1
        y_online2, z2 = reps2

        # enqueue the new representation.
        self._dequeue_and_enqueue(torch.cat([z1,z2], dim=0))

        if not self.queue_full:
            print(f"filling up the queue.")
            return 0*torch.norm(z1) # not generating any loss.

        # if singular then cut the dimension by the half of the rank.
        q_rank = torch.matrix_rank(torch.mm(self.queue.T, self.queue))
        if q_rank == self.queue.shape[1]:
            w = _whiten(self.queue, eps=self.eps, whiten_method=self.whiten) # symmetric whitening matrix
            w_z1, w_z2 = torch.mm(z1, w), torch.mm(z2, w)
        else:
            w_fs = int(q_rank // 2)
            perm = torch.randperm(z1.shape[1])
            z1, z2, q = z1[:, perm][:, :w_fs], z2[:, perm][:,:w_fs], self.queue[:,perm][:,:w_fs]

            w = _whiten(q, eps=self.eps, whiten_method=self.whiten)
            w_z1, w_z2 = torch.mm(z1, w), torch.mm(z2, w)

        loss = 0.5 * ((self.normalize(z1) - self.normalize(w_z2)).square().mean() +
                       (self.normalize(z2) - self.normalize(w_z1)).square().mean())

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
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        bs = keys.shape[0]

        ptr = int(self.queue_ptr)
        if ptr + bs >= self.queue_size:
            self.queue_full = True
        if self.queue_size % bs != 0:
            print(self.queue_size, bs)
        assert self.queue_size % bs == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + bs, :] = keys
        ptr = (ptr + bs) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

#######################################
# utils
# whitening related
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
        return torch.inverse(L).T
    else:
        raise NotImplementedError(f"{whiten_method} is not implemented yet.")


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


if __name__ == '__main__':
    model = Model()
    print(model)