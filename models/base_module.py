import os

import torch
from torch import nn


class BaseModule(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.__name = name

    def get_name(self) -> str:
        return self.__name

    def save(self, root):
        path = os.path.join(root, self.get_name())
        torch.save(self.state_dict(), path)
