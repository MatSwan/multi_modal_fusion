import os
from typing import List

import torch
from torch import nn

from models.base_module import BaseModule


class BaseExperiment(nn.Module):
    def __init__(self, root):
        self.root: str = root
        super().__init__()

    def save(self, models):
        for model in models:
            for module in model:
                path = os.path.join(self.root, module.get_name())
                torch.save(module.state_dict(), path)

