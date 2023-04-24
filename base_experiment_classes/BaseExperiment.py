import os

import torch
from torch import nn


class BaseExperiment(nn.Module):
    def __init__(self, root, experiment_name):
        self.root: str = os.sep.join([root, experiment_name])
        super().__init__()

    def save(self, models, save_path=None, recursive=False):
        save_path = self.root if save_path is None else save_path
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        for model in models:
            if recursive and hasattr(model, '__iter__'):  # this is scary
                self.save(models=model, save_path=save_path, recursive=True)
                continue
            path = os.sep.join([save_path, model.get_name()]) + '.pth'
            torch.save(model.state_dict(), path)

    # def save(self, models, trial_number:str = None, save_path=None):
    #     save_path = self.save_path if save_path is None else save_path
    #     for model in models:
    #         if isinstance(model, nn.ModuleList):
    #             self.save(model)
    #             continue
    #         if trial_number is not None:
    #             save_root = os.sep.join([save_path, f'models_for_trial_number_{trial_number}'])
    #         else: save_root = save_path
    #         if not os.path.isdir(save_root):
    #             os.mkdir(save_root)
    #         path = os.sep.join([save_root, model.get_name()]) + '.pth'
    #         torch.save(model.state_dict(), path)
