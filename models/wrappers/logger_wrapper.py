import numpy as np
import torch

import collections

import warnings

from models import WRAPPERS
from models.wrappers import BaseWrapper

import scipy.spatial as spatial

from torch.nn import functional as F


# Logger is outside of the weight wrapper,
# so it should support the update() method in weight wrapper

@WRAPPERS.register_module('emptylogger')
class EmptyLogger(BaseWrapper):
    def __init__(self, model, **kwargs):
        super(EmptyLogger, self).__init__(model=model)

        # abling the multi-layer logger wrappers
        if hasattr(self.model, 'metrics'):
            self.metrics = self.model.metrics
        else:
            self.metrics = collections.defaultdict(list)    # can access the item without initialization

    @torch.no_grad()
    def estimate(self):
        """
        Provide basic checking on the module.reps
        :return:
        """
        if not hasattr(self.module, 'reps'):
            warnings.warn(f"model '{str(self.module)}' doesn't have attribute 'reps', disabled alignment estimation", RuntimeWarning)

            self.estimate = _empty_func

        # multi-layer logger-wrappers
        if isinstance(self.model, EmptyLogger):
            self.model.estimate()

    def clear(self):
        self.metrics.clear()

    def update(self):
        self.model.update()


@WRAPPERS.register_module('alignlogger')
class AlignLogger(EmptyLogger):
    def __init__(self, model, **kwargs):
        super(AlignLogger, self).__init__(model=model)

    @torch.no_grad()
    def estimate(self):
        super(AlignLogger, self).estimate()

        reps = self.module.reps
        # parse names
        views = collections.defaultdict(list)
        for key in reps:
            name, _ = _parse_name(key)
            views[name].append(reps[key])

        for key, lst in views.items():
            # only estimate the first two views
            v1, v2 = lst[0], lst[1]
            normed_v1, normed_v2 = F.normalize(v1 - v1.mean(0)), F.normalize(v2 - v2.mean(0))
            self.metrics[f'align.{key}'].append(_lalign(v1, v2))
            self.metrics[f'align_normed.{key}'].append(_lalign(normed_v1, normed_v2))

        # try:
        #     for key, lst in views.items():
        #         # only estimate the first two views
        #         v1, v2 = lst[0], lst[1]
        #         normed_v1, normed_v2 = F.normalize(v1 - v1.mean(0)), F.normalize(v2 - v2.mean(0))
        #         self.metrics[f'align.{key}'].append(_lalign(v1, v2))
        #         self.metrics[f'align_normed.{key}'].append(_lalign(normed_v1, normed_v2))
        # except IndexError:
        #     warnings.warn('index out of range, disable estimation.', RuntimeWarning)


@WRAPPERS.register_module('uniformlogger')
class UniformLogger(EmptyLogger):
    def __init__(self, model, **kwargs):
        super(UniformLogger, self).__init__(model=model)

    @torch.no_grad()
    def estimate(self):
        # only estimate the online networks' uniformity
        super(UniformLogger, self).estimate()

        reps = self.module.reps
        # parse names
        views = collections.defaultdict(list)
        for key in reps:
            name, _ = _parse_name(key)
            if 'online' in name:
                views[name].append(reps[key])

        for key, lst in views.items():
            # only estimate the first views
            vec = lst[0]
            vec = F.normalize(vec - vec.mean(0))
            try:
                uniform = _lunif_gpu(vec)  # torch.pdist is not supported by apex.amp
            except:
                uniform = _lunif_cpu(vec)
            self.metrics[f'uniform.{key}'].append(uniform)

        # try:
        #     for key, lst in views.items():
        #         # only estimate the first views
        #         vec = lst[0]
        #         vec = F.normalize(vec - vec.mean(0))
        #         try:
        #             uniform = _lunif_gpu(vec)  # torch.pdist is not supported by apex.amp
        #         except:
        #             uniform = _lunif_cpu(vec)
        #         self.metrics[f'uniform.{key}'].append(uniform)
        # except IndexError:
        #     warnings.warn('UniformLogger index out of range, disable estimation.', RuntimeWarning)



@WRAPPERS.register_module('prednormlogger')
class PredNormLogger(EmptyLogger):
    def __init__(self, model, **kwargs):
        super(PredNormLogger, self).__init__(model=model)

    @torch.no_grad()
    def estimate(self):
        super(PredNormLogger, self).estimate()
        try:
            for name, param in self.module.pred.named_parameters():
                self.metrics[f'norm.pred.{name}'].append(torch.norm(param).detach().item())
        except :
            warnings.warn('model doesnt have pred property.')


@WRAPPERS.register_module('repnormlogger')
class RepNormLogger(EmptyLogger):
    def __init__(self, model, **kwargs):
        super(RepNormLogger, self).__init__(model=model)

    @torch.no_grad()
    def estimate(self):
        super(RepNormLogger, self).estimate()

        reps = self.module.reps

        for key, val in reps.items():
            # only estimate the first views
            norm = torch.norm(val) / val.shape[0]
            self.metrics[f'rep_norm.{key}'].append(norm.detach().item())


# utils
def _parse_name(name_str):
    lst = name_str.split('.')
    view = lst[-1]
    name = '.'.join(lst[:-1])
    return name, view


def _empty_func(*input):
    pass


def _lalign(x1, x2):
    return (x1 - x2).square().sum(dim=0).mean().detach().item()


def _lunif_cpu(x, t=2):
    x = x.cpu().numpy()
    # sq_pdist = torch.pdist(x, p=2).pow(2)     # not supported in AMP
    sq_pdist = torch.Tensor(spatial.distance.pdist(x, 'minkowski', p=2)).pow(2)
    return sq_pdist.mul(-t).exp().mean().log()


def _lunif_gpu(x, t=2):
    # sq_pdist = torch.pdist(x, p=2).pow(2)     # not supported in AMP
    sq_pdist = torch.pdist(x, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean().log().detach().item()