import numpy as np
import torch
import torch.nn as nn

class Ema(object):
    def __init__(self, target, target_proj, opt='sgd', lr=4e-3, momentum=0.9):
        super(Ema, self).__init__()
        assert opt in ('sgd', 'adam', 'momentum', 'rmsprop')

        self.target, self.target_proj = target, target_proj

        params = list(target.parameters())+list(target_proj.parameters())

        if opt == 'adam':
            self.optimizer = torch.optim.Adam(lr=lr, params=params)
        elif opt == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(lr=lr, params=params)
        elif opt == 'momentum':
            self.optimizer = torch.optim.SGD(lr=lr, params=params, momentum=momentum)
        else:
            self.optimizer = torch.optim.SGD(lr=lr, params=params)

    def update(self, online, online_proj):
        """
        Need to provide the new online parameters for the EMA update.
        :param online:
        :param online_proj:
        :return:
        """
        with torch.no_grad():
            for target_param, online_param in [
                *zip(self.target.parameters(), online.parameters()),
                *zip(self.target_proj.parameters(), online_proj.parameters())
            ]:
                target_param.grad = (target_param.data - online_param.data)

            self.optimizer.step()