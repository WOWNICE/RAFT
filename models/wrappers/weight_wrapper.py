import numpy as np
import torch
import torch.nn as nn
import collections

class EmptyWrapper(nn.Module):
    def __init__(self, model, **kwargs):
        super(EmptyWrapper, self).__init__()
        self.model = model

    def forward(self, x1, x2):
        return self.model(x1, x2)

    def estimate_align(self):
        return self.model.estimate_align()

    def estimate_uniform(self):
        return self.model.estimate_uniform()

    def estimate_cross(self):
        return self.model.estimate_cross()

    def update(self):
        pass

class EmaWrapper(nn.Module):
    def __init__(self, model, opt='sgd', lr=4e-3, momentum=0.9, **kwargs):
        super(EmaWrapper, self).__init__()
        self.model = model

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

    def forward(self, x1, x2):
        return self.model(x1, x2)

    def estimate_align(self):
        return self.model.estimate_align()

    def estimate_uniform(self):
        return self.model.estimate_uniform()

    def estimate_cross(self):
        return self.model.estimate_cross()

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


class SlowRandPredWrapper(EmaWrapper):
    def __init__(self, model, opt='sgd', lr=4e-3, momentum=0.9, **kwargs):
        super(SlowRandPredWrapper, self).__init__(
            model=model,
            opt=opt,
            lr=lr,
            momentum=momentum
        )

    def update_pred(self):
        def reset_param(layer):
            if hasattr(layer, 'reset_parameters'):
                dic = layer.state_dict()
                layer.reset_parameters()
                for name, param in layer.named_parameters():
                    param.data = 0.004*param.data + 0.996*dic[name].data

        self.model.pred.apply(reset_param)

    def update(self):
        super(SlowRandPredWrapper, self).update()
        self.update_pred()
