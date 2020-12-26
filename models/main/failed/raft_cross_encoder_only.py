import numpy as np
import torch
import torch.nn as nn

from kornia import augmentation as augs
from kornia import filters, color

# mapping from the model name to the model constructor
from models import name_model_dic as func_dic
from models.main import RandomApply

import scipy.spatial as spatial

from models.main.byol import Model as Byol


class Model(Byol):
    """
    RAFT model. only the alignment loss is computed right after the projector but not predictor.
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
            # losses
            alignment_weight=1.,
            cross_weight=1.,
            same_init=False,
            **kwargs
    ):
        super(Model, self).__init__(
            encoder=encoder,
            projector=projector,
            predictor=predictor,
            normalization=normalization,
            # shapes
            input_shape=input_shape,
            proj_shape=proj_shape,
            predictor_shape=predictor_shape,
            same_init=same_init,
            # losses
        )

        self.alignment_weight = alignment_weight
        self.cross_weight = cross_weight

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
        # z_online1 = self.normalize(z_online1)
        # z_online2 = self.normalize(z_online2)
        z_target1 = self.normalize(z_target1)
        z_target2 = self.normalize(z_target2)

        # use cross-model loss to constrain the predictor only.
        z_predonly1 = self.normalize(self.pred(z_online1.detach()))
        z_predonly2 = self.normalize(self.pred(z_online2.detach()))

        # compute the loss
        loss_align = (z_online_pred1 - z_online_pred2).square().mean()
        loss_cross_predonly = ((z_predonly1 - z_target1).square().mean() + (z_predonly2 - z_target2).square().mean()) / 2
        loss_cross = ((z_online_pred1 - z_target1).square().mean() + (z_online_pred2 - z_target2).square().mean()) / 2
        loss = self.alignment_weight * loss_align - self.cross_weight * (loss_cross - loss_cross_predonly)

        return loss


if __name__ == '__main__':
    model = Model()
    print(model)