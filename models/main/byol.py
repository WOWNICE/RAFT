import numpy as np
import torch
import torch.nn as nn

from kornia import augmentation as augs
from kornia import filters, color

# mapping from the model name to the model constructor
from models import name_model_dic as func_dic
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
            projector='byol-proj',
            predictor='byol-proj',
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
        encoder, projector, predictor, normalization = func_dic[encoder], func_dic[projector], func_dic[predictor], func_dic[normalization]

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

        # set the transformations
        self._set_transforms(input_shape[1])

    def gen_rep(self, x):
        view = self.train_augment(x)
        y_online = self.online(view)
        z_online = self.online_proj(y_online)
        z_online_pred = self.pred(z_online)

        with torch.no_grad():
            y_target = self.target(view)
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
    def register_reps(self, reps1, reps2):
        """
        register the representations to the model
        :param reps1:
        :param reps2:
        :return:
        """
        y_online1, z_online1, z_online_pred1, y_target1, z_target1 = reps1
        y_online2, z_online2, z_online_pred2, y_target2, z_target2 = reps2

        # for estimate uniformity/alignment/others in the future
        # to save the computational cost, we only preserve up to 64 samples

        batch_size = min(y_online2.shape[0], 64)
        self.z_online1 = self.normalize(z_online1)[:batch_size,:]
        self.z_online2 = self.normalize(z_online2)[:batch_size,:]
        self.z_online_pred1 = self.normalize(z_online_pred1)[:batch_size,:]
        self.z_online_pred2 = self.normalize(z_online_pred2)[:batch_size,:]
        self.z_target1 = self.normalize(z_target1)[:batch_size,:]
        self.z_target2 = self.normalize(z_target2)[:batch_size,:]

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
        align_pred = (self.z_online_pred1 - self.z_online_pred2).square().mean().detach()
        align_target = (self.z_target1 - self.z_target2).square().mean().detach()

        return align_online, align_pred, align_target

    @ torch.no_grad()
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

        uniform_online = (lunif(self.z_online1) + lunif(self.z_online2)) / 2
        uniform_pred = (lunif(self.z_online_pred1) + lunif(self.z_online_pred2)) / 2
        uniform_target = (lunif(self.z_target1) + lunif(self.z_target2)) / 2

        return uniform_online, uniform_pred, uniform_target

    @torch.no_grad()
    def estimate_cross(self):
        """
        the cross-model loss between online-target/pred-target.
        :return:
        """
        cross_online = ((self.z_online1 - self.z_target1).square().mean() + (self.z_online2 - self.z_target2).square().mean()) / 2
        cross_pred = ((self.z_online_pred1 - self.z_target1).square().mean() + (self.z_online_pred2 - self.z_target2).square().mean()) / 2

        return cross_online, cross_pred

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