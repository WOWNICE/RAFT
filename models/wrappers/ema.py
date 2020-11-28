import numpy as np
import torch
import torch.nn as nn

class EmaWrapper(nn.Module):
    def __init__(self, model, opt='sgd', lr=4e-3, momentum=0.9):
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