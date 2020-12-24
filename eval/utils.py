import numpy as np
import torch
import torch.nn as nn
import collections
from models import *

dataset_classes = {
    'cifar10':      10,
    'cifar100':     100,
    'subimagenet': 100,
    'imagenet': 1000,
}

def correct_k(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res


def load_model(model_checkpoint, encoder='resnet50', online=True, projector='', predictor=''):
    # load the evaluate
    dic = torch.load(model_checkpoint, map_location=torch.device('cpu'))

    # load the online/target param in a new state dic
    online_param = collections.OrderedDict()
    target_param = collections.OrderedDict()

    for k, v in dic.items():
        if 'online.' in k:
            online_param[k[7:]] = v
        elif 'target.' in k:
            target_param[k[7:]] = v

    encoder = ENCODERS[encoder]()
    if online:
        encoder.load_state_dict(online_param)
    else:
        encoder.load_state_dict(target_param)

    components = [encoder]
    # TODO: support evaluation on projector & predictor
    # if projector:
    #     model.append(name_model_dic[projector]())

    model = nn.Sequential(*components)
    return model