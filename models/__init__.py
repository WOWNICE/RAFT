import torch.nn.functional as F
from torch import nn

from utils.registry import Registry

__all__ = ['WRAPPERS', 'HEADS', 'ENCODERS', 'FUNCS']


WRAPPERS = Registry('wrapper')
HEADS = Registry('heads')
ENCODERS = Registry('encoders')
FUNCS = Registry('functions')

FUNCS.register_module('l2')(F.normalize)
FUNCS.register_module('I')(nn.Identity())

# whenever creating a new module, register it here.
from .submodels.resnets import *
from .submodels.mlps import *
from .wrappers.weight_wrapper import *