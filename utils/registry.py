import numpy as np
import torch
import torch.nn as nn
import inspect


class Registry(object):
    def __init__(self, name):
        assert isinstance(name, str)
        self.name = name
        self.module_dict = {}

    def __str__(self):
        return f'registry name: {self.name}; \n registry dict: {self.module_dict.__str__()}'

    def __getitem__(self, key):
        return self.module_dict[key]

    def register_module(self, name=None):
        def add(name, cls):

            if not callable(cls):
                raise TypeError(f'module to be registered should be a class or function, but got {type(cls)}.')

            if name is None:
                name = cls.__name__

            if name in self.module_dict:
                raise KeyError(f'module {name} already registered.')

            self.module_dict[name] = cls

            return cls


        return lambda x: add(name, x)


if __name__ == '__main__':
    reg = Registry('test')
