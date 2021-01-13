import numpy as np
import torch
import torch.nn as nn
import collections

from models import WRAPPERS
from models.wrappers import BaseWrapper

@WRAPPERS.register_module('empty')
class EmptyWrapper(BaseWrapper):
    def __init__(self, model, **kwargs):
        super(EmptyWrapper, self).__init__(model=model)

    def forward(self, *input):
        loss = self.model(*input)
        self.update()
        return loss

    def update(self):
        pass


@WRAPPERS.register_module('ema')
class EmaWrapper(EmptyWrapper):
    def __init__(self, model, opt='sgd', lr=4e-3, momentum=0.9, **kwargs):
        super(EmaWrapper, self).__init__(model=model)

        params = list(model.target.parameters())+list(model.target_proj.parameters())

        self.opt = None
        if opt in ['fix', 'copy']:
            self.opt = opt
        elif opt == 'adam':
            self.optimizer = torch.optim.Adam(lr=lr, params=params)
        elif opt == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(lr=lr, params=params)
        elif opt == 'momentum':
            self.optimizer = torch.optim.SGD(lr=lr, params=params, momentum=momentum)
        else:
            self.optimizer = torch.optim.SGD(lr=lr, params=params)

    def update(self):
        # if ema is included
        if not self.opt:
            with torch.no_grad():
                for target_param, online_param in [
                    *zip(self.model.target.parameters(), self.model.online.parameters()),
                    *zip(self.model.target_proj.parameters(), self.model.online_proj.parameters())
                ]:
                    target_param.grad = (target_param.data - online_param.data)

                self.optimizer.step()

        elif self.opt == 'copy':
            with torch.no_grad():
                for target_param, online_param in [
                    *zip(self.model.target.parameters(), self.model.online.parameters()),
                    *zip(self.model.target_proj.parameters(), self.model.online_proj.parameters())
                ]:
                    target_param.data = online_param.data

        elif self.opt == 'fix':
            pass

@WRAPPERS.register_module('emacosine')
class EmaCosineWrapper(EmaWrapper):
    def __init__(self, model, opt='sgd', lr=4e-3, momentum=0.9, K=200, k=0, **kwargs):
        super(EmaCosineWrapper, self).__init__(
            model=model,
            opt=opt,
            lr=lr,
            momentum=momentum
        )
        self.K = K
        self.k = k
        self.lr =lr

    def update(self):
        super(EmaCosineWrapper, self).update()

        # update the epoch number k
        self.k += 1

        # compute new lr
        new_lr = self.lr * (np.cos(np.pi*self.k/self.K)+1)/2    # equivalent form of the original BYOL paper.

        # update the optimizer learning
        for g in self.optimizer.param_groups:
            g['lr'] = new_lr


@WRAPPERS.register_module('freezepred')
class FreezePredWrapper(EmaWrapper):
    def __init__(self, model, opt='sgd', lr=4e-3, momentum=0.9, **kwargs):
        super(FreezePredWrapper, self).__init__(
            model=model,
            opt=opt,
            lr=lr,
            momentum=momentum
        )

        for param in self.model.pred.parameters():
            param.requires_grad = False


@WRAPPERS.register_module('optimalpred')
class OptimalPredWrapper(FreezePredWrapper):
    def __init__(self, model, opt='sgd', lr=4e-3, momentum=0.9, pred_path='', **kwargs):
        super(OptimalPredWrapper, self).__init__(
            model=model,
            opt=opt,
            lr=lr,
            momentum=momentum
        )

        # load the optimal predictor and fix
        if pred_path:
            dic = torch.load(pred_path, map_location=torch.device('cpu'))
            pred_dic = collections.OrderedDict()
            for k, v in dic.items():
                if 'pred' in k:
                    pred_dic[k[5:]] = v

            self.model.pred.load_state_dict(pred_dic)


@WRAPPERS.register_module('randpred')
class RandPredWrapper(EmaWrapper):
    def __init__(self, model, opt='sgd', lr=4e-3, momentum=0.9, **kwargs):
        super(RandPredWrapper, self).__init__(
            model=model,
            opt=opt,
            lr=lr,
            momentum=momentum
        )

    def update_pred(self):
        def reset_param(layer):
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        self.model.pred.apply(reset_param)

    def update(self):
        super(RandPredWrapper, self).update()
        self.update_pred()

