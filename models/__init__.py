import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



from models.submodels.resnets import resnet18, resnet34, resnet50

from models.submodels.mlps import *

name_model_dic = {
    # encoder networks
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,

    # projector and predictors
    'byol-proj': ProjectorByol,
    'byol-proj-nobn': ProjectorByolNoBN,
    'simclr-proj': ProjectorSimClr,
    'linear-proj': ProjectorLinear,
    'linear-proj-nobn': ProjectorLinearNoBN,
    'identity-proj': nn.Identity,

    # emas

    # normalization methods
    'l2':       F.normalize,
    'I':        nn.Identity()
}