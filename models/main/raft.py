import numpy as np
import torch
import torch.nn as nn


# mapping from the model name to the model constructor

from models.main.byol import Model as Byol

from models import *


class Model(Byol):
    """
    RAFT model.
    """
    def __init__(
            self,
            # models
            encoder='resnet50',
            projector='2layermlpbn',
            predictor='2layermlpbn',
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
        z_target1 = self.normalize(z_target1)
        z_target2 = self.normalize(z_target2)

        # compute the loss
        bs = z_online_pred1.shape[0]
        loss_align = 2 - 2 * (z_online_pred1 * z_online_pred2).sum() / bs
        loss_cross = 2 - (z_online_pred1 * z_target1 + z_online_pred2 * z_target2).sum() / bs
        loss = self.alignment_weight * loss_align - self.cross_weight * loss_cross

        return loss


if __name__ == '__main__':
    model = Model()
    print(model)