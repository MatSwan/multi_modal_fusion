from __future__ import annotations

import os
from math import floor
from typing import Any, Optional, TypeVar, List

import torch
from torch import nn, Tensor

class ActivateGrad:
    def __init__(self, *modules: nn.Module) -> None:
        self._modules = modules

    def __enter__(self) -> None:
        set_requires_grad(True, *self._modules)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        set_requires_grad(False, *self._modules)


class DeactivateGrad:
    def __init__(self, *modules: nn.Module) -> None:
        self._modules = modules

    def __enter__(self) -> None:
        set_requires_grad(False, *self._modules)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        set_requires_grad(True, *self._modules)


def build_signature(*args: Any) -> str:
    output = ""
    if len(args) != 0:
        for arg in args[0:-1]:
            output += str(arg) + 'x'
        output += str(args[-1])
    if len(output) > 0: output = "_" + output
    return output


T = TypeVar("T", bound=nn.Module)


def load_network(model: T, path: str) -> T:
    model.load_state_dict(torch.load(path))
    return model


def save_network(model: T, path: str) -> T:
    root, _ = os.path.split(path)
    if not os.path.isdir(root): os.makedirs(root)
    torch.save(model.state_dict(), path)
    return model


def set_requires_grad(flag: bool, *modules: T) -> Optional[T]:
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = flag
    if len(modules) == 1: return modules[0]


def weights_init_normal(model: nn.Module) -> None:
    class_name = model.__class__.__name__
    if class_name.find('Conv') != -1:
        torch.nn.init.normal_(model.weight, 0.0, 0.02)
    elif class_name.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(model.weight, 1.0, 0.02)
        torch.nn.init.constant_(model.bias, 0.0)


def update_size(in_size: int, kernel_size: int, *, stride: int = 1, padding: int = 0, dilation: int = 1) -> int:
    return floor((in_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)


def detach_list(tensor_list: List[Tensor]) -> List[Tensor]: return [x.detach() for x in tensor_list]
