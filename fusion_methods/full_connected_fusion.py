from typing import List

import torch
from torch import nn, Tensor

class FusionMethod(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, inputs: List[Tensor]) -> Tensor:
        return super().__call__(inputs)


class FullyConnectedFusion(FusionMethod):
    def __init__(self, number_of_modalities: int, vectorized_feature_length: int) -> None:
        super().__init__()
        self.model = nn.Sequential(self.create_fusion_layer(number_of_modalities=number_of_modalities,
                                                            vectorized_feature_length=vectorized_feature_length))

    def create_moddel(self, number_of_modalities: int, vectorized_feature_length: int) -> None:
        self.model = self.create_fusion_layer(number_of_modalities=number_of_modalities)

    def create_fusion_layer(self, number_of_modalities, vectorized_feature_length):
        return nn.Linear(vectorized_feature_length * number_of_modalities, vectorized_feature_length)

    def forward(self, inputs: List[Tensor]) -> Tensor:
        inputs = torch.cat(tuple(inputs), dim=1)
        return self.models(inputs)


class FullyConnectedFusionFactory:
    "Factories server as a namespace for pretrain options and when called will return an instance of the approeriate fusion method"
    def __init__(self, pretrain:bool = False, pretrain_epoch:int = 0, freeze_features: bool = False ) -> None:
        self.pretrain = pretrain
        self.freeze_features = freeze_features
        if not self.pretrain:
            self.pretrain_epoch = 0
        else: self.pretrain_epoch = pretrain_epoch
        self.name = f'fc_fusion_pt{pretrain}'


    def __call__(self, number_of_modalities: int , vectorized_feature_length:int) -> FullyConnectedFusion:
        return FullyConnectedFusion(number_of_modalities=number_of_modalities, vectorized_feature_length=vectorized_feature_length)


