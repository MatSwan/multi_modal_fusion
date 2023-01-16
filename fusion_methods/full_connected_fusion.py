from typing import List

import torch
from torch import nn, Tensor

class FusionMethod(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, inputs: List[Tensor]) -> Tensor:
        return super().__call__(inputs)


class FullyConnectedFusion(FusionMethod):
    def __init__(self, number_of_modalities: int, vectorized_feature_length) -> None:
        super().__init__()
        self.model = nn.Sequential(self.create_fusion_layer(number_of_modalities=number_of_modalities,
                                                            vectorized_feature_length=vectorized_feature_length))

    def create_fusion_layer(self, number_of_modalities, vectorized_feature_length):
        return nn.Linear(vectorized_feature_length * number_of_modalities, vectorized_feature_length)

    def forward(self, inputs: List[Tensor]) -> Tensor:
        inputs = torch.cat(tuple(inputs), dim=1)
        return self.models(inputs)
