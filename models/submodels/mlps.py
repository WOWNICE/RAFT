import torch.nn as nn

from models import HEADS

@HEADS.register_module('linearbn')
class ProjectorLinear(nn.Module):
    def __init__(self, shape=()):
        super(ProjectorLinear, self).__init__()
        if len(shape) < 2:
            raise Exception("Wrong shape for Projector")

        self.main = nn.Sequential(
            nn.Linear(shape[0], shape[-1]),
            nn.BatchNorm1d(shape[-1]),
        )

    def forward(self, x):
        return self.main(x)

@HEADS.register_module('linear')
class ProjectorLinearNoBN(nn.Module):
    def __init__(self, shape=()):
        super(ProjectorLinearNoBN, self).__init__()
        if len(shape) < 2:
            raise Exception("Wrong shape for Projector")

        self.main = nn.Linear(shape[0], shape[-1])

    def forward(self, x):
        return self.main(x)

@HEADS.register_module('2layermlp')
class ProjectorByolNoBN(nn.Module):
    def __init__(self, shape=()):
        super(ProjectorByolNoBN, self).__init__()
        if len(shape) < 3:
            raise Exception("Wrong shape for Projector")

        self.main = nn.Sequential(
            nn.Linear(shape[0], shape[1]),
            nn.ReLU(),
            nn.Linear(shape[1], shape[2])
        )

    def forward(self, x):
        return self.main(x)

@HEADS.register_module('2layermlpbn')
class ProjectorByol(nn.Module):
    """
    Default BYOL projector
    """
    def __init__(self, shape=()):
        super(ProjectorByol, self).__init__()
        if len(shape) < 3:
            raise Exception("Wrong shape for Projector")

        self.main = nn.Sequential(
            nn.Linear(shape[0], shape[1]),
            nn.BatchNorm1d(shape[1]),
            nn.ReLU(),
            nn.Linear(shape[1], shape[2])
        )

    def forward(self, x):
        return self.main(x)