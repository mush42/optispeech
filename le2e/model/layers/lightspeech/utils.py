import math
from collections import defaultdict

import numpy as np
import copy

import torch
import torch.nn as nn


def build_activation(act_func, inplace=True):
    if act_func == 'relu':
        return nn.ReLU(inplace=inplace)
    elif act_func == 'relu6':
        return nn.ReLU6(inplace=inplace)
    elif act_func == 'gelu':
        return GeLU()
    elif act_func == 'gelu_accurate':
        return GeLUAcc()
    elif act_func == 'tanh':
        return nn.Tanh()
    elif act_func == 'sigmoid':
        return nn.Sigmoid()
    elif act_func is None:
        return None
    else:
        raise ValueError('do not support: %s' % act_func)

def make_positions(tensor, padding_idx):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (
                   torch.cumsum(mask, dim=1).type_as(mask) * mask
           ).long() + padding_idx


def softmax(x, dim):
    return F.softmax(x, dim=dim, dtype=torch.float32)


def sequence_mask(lengths, maxlen, dtype=torch.bool):
    if maxlen is None:
        maxlen = lengths.max()
    mask = ~(torch.ones((len(lengths), maxlen)).to(lengths.device).cumsum(dim=1).t() > lengths).t()
    mask.type(dtype)
    return mask


INCREMENTAL_STATE_INSTANCE_ID = defaultdict(lambda: 0)


def _get_full_incremental_state_key(module_instance, key):
    module_name = module_instance.__class__.__name__

    # assign a unique ID to each module instance, so that incremental state is
    # not shared across module instances
    if not hasattr(module_instance, '_instance_id'):
        INCREMENTAL_STATE_INSTANCE_ID[module_name] += 1
        module_instance._instance_id = INCREMENTAL_STATE_INSTANCE_ID[module_name]

    return '{}.{}.{}'.format(module_name, module_instance._instance_id, key)


def get_incremental_state(module, incremental_state, key):
    """Helper for getting incremental state for an nn.Module."""
    full_key = _get_full_incremental_state_key(module, key)
    if incremental_state is None or full_key not in incremental_state:
        return None
    return incremental_state[full_key]


def set_incremental_state(module, incremental_state, key, value):
    """Helper for setting incremental state for an nn.Module."""
    if incremental_state is not None:
        full_key = _get_full_incremental_state_key(module, key)
        incremental_state[full_key] = value


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


def fill_with_neg_inf2(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(-1e9).type_as(t)
