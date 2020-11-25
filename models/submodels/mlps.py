import numpy as np
import torch
import torch.nn as nn

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


class ProjectorLinearNoBN(nn.Module):
    def __init__(self, shape=()):
        super(ProjectorLinearNoBN, self).__init__()
        if len(shape) < 2:
            raise Exception("Wrong shape for Projector")

        self.main = nn.Linear(shape[0], shape[-1])

    def forward(self, x):
        return self.main(x)


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

class ProjectorByol(nn.Module):
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

class ProjectorSimClr(nn.Module):
    """
    Additional BN layer after the last linear
    """
    def __init__(self, shape=()):
        super(ProjectorSimClr, self).__init__()
        if len(shape) < 3:
            raise Exception("Wrong shape for Projector")

        self.main = nn.Sequential(
            nn.Linear(shape[0], shape[1]),
            nn.BatchNorm1d(shape[1]),
            nn.ReLU(),
            nn.Linear(shape[1], shape[2]),
            nn.BatchNorm1d(shape[2])
        )

    def forward(self, x):
        return self.main(x)