import torch.nn.functional as F
from torch import nn

from utils.registry import Registry

__all__ = ['WRAPPERS', 'HEADS', 'ENCODERS', 'FUNCS', 'AUGS']


WRAPPERS = Registry('wrapper')
HEADS = Registry('heads')
ENCODERS = Registry('encoders')
FUNCS = Registry('functions')
AUGS = Registry('augmentations')

FUNCS.register_module('l2')(F.normalize)
FUNCS.register_module('I')(nn.Identity())

# whenever creating a new module, register it here.
from .submodels.resnets import *
from .submodels.mlps import *
from .submodels.augmentations import *
from .wrappers.weight_wrapper import *
from .wrappers.logger_wrapper import *
from .wrappers.lr_wrapper import *