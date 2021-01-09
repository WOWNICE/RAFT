import torch
import numpy as np

from torch import nn
from torch import optim

from models import WRAPPERS

@WRAPPERS.register_module('lr.empty')
class BaseLRWrapper(object):
    def __init__(self, optimizer, baselr=None, **kwargs):
        super(BaseLRWrapper, self).__init__()
        assert isinstance(optimizer, torch.optim.Optimizer)
        self.optimizer = optimizer

        self.baselr = baselr if baselr else optimizer.param_groups[0]['lr']

    def step(self):
        self.optimizer.step()
        self.update_lr()

    def update_lr(self):
        pass

    def zero_grad(self):
        self.optimizer.zero_grad()


class BaseWarmupAnnealing(BaseLRWrapper):
    """
    Base class for the warmup and annealing lr_scheduler.
    max_lr = baselr
    min_lr = baselr
    """
    def __init__(
            self,
            optimizer,
            total_steps=None, total_epochs=None, steps_per_epoch=None,
            warmup_steps=None, warmup_epochs=None,
            warm_anneal=(None, None),
            baselr=None,
            c_step=None
    ):
        super(BaseWarmupAnnealing, self).__init__(optimizer, baselr)

        # total steps
        if total_steps is None and (total_epochs is None or steps_per_epoch is None):
            raise ValueError('Enough info needs to be given to compute the total training steps.')
        self.total_steps = total_steps if total_steps else total_epochs * steps_per_epoch

        # warmup steps
        if warmup_steps is None and (warmup_epochs is None or steps_per_epoch is None):
            raise ValueError('Enough info needs to be given to compute the warmup training steps.')
        self.warmup_steps = warmup_steps if warmup_steps else warmup_steps * steps_per_epoch

        self.c_step = c_step if c_step else 0
        self.max_lr = self.baselr
        self.min_lr = self.baselr * 1e-3

        self.warmup, self.anneal = warm_anneal


    def step(self):
        self.optimizer.step()
        self.update_lr()
        self.c_step += 1

    def update_lr(self):
        if self.c_step < self.warmup_steps:
            lr = self.warmup(
                step=self.c_step,
                max_lr=self.max_lr,
                min_lr=self.min_lr,
                warmup_steps=self.warmup_steps
            )
        else:
            lr = self.anneal(
                step=self.c_step-self.warmup_steps,
                max_lr=self.max_lr,
                min_lr=self.min_lr,
                total_steps=self.total_steps-self.warmup_steps
            )

        # update the learning rate in optimizer.
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        print(lr)


# define several warmup functions and anneal functions
def warmup_linear(step, max_lr, min_lr, warmup_steps):
    return min_lr + (max_lr - min_lr) * step / warmup_steps

def anneal_cosine(step, max_lr, min_lr, total_steps):
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(step * np.pi / total_steps))


@WRAPPERS.register_module('lr.linearcosine')
def linearcosine(*args, **kwargs):
    return BaseWarmupAnnealing(*args, warm_anneal=(warmup_linear, anneal_cosine), **kwargs)


