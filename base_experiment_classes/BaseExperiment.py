import os
from typing import List

import torch

from models.base_module import BaseModule


class BaseExperiment():
    def __init(self, root ):
        self.root: str = root

    def save(self, models):
        for model in models:
            for module in model:
                path = os.path.join(self.root, module.get_name())
                torch.save(module.state_dict(), path)

