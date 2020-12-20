# Wrapper Design Principle
# if the model has the attribute "module", then create a link to this module,
# this is for supporting multi-layer wrappers in the future
# module is the core nn.Module component that defines the crucial computation

from torch import nn

class BaseWrapper(nn.Module):
    def __init__(self, model, **kwargs):
        super(BaseWrapper, self).__init__()
        self.model = model
        if hasattr(model, 'module'):
            self.module = model.module
        else:
            self.module = model # first layer wrapper

    def forward(self, *input):
        return self.model(*input)