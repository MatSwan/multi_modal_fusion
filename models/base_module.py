import os

import torch
from torch import nn


class BaseModule(nn.Module):
    def __init__(self, name: str):
        self.__name = name
        super().__init__()

    def get_name(self) -> str:
        return self.__name

    def save(self, root):
        path = os.path.join(root, self.get_name())
        torch.save(self.state_dict(), path)
