from typing import List, Callable

import torch
from torch import Tensor, nn

from models.base_module import BaseModule


class Vgg16PreFusion(BaseModule):
    def __init__(self, name: str, channel:int, max_channel:int,  norm: Callable[[int], nn.Module] = nn.BatchNorm2d,
                 activation: Callable[[], nn.Module] = lambda: nn.ReLU(inplace=True)
                 ) -> None:
        super().__init__(name=name)

        def create_model(feature_list: List[int], channel_size_list: List[int], input_channel, pool):
            model = []
            previous_feature = input_channel
            for feature, channel_size in zip(feature_list, channel_size_list):
                for feature_number in range(feature):
                    if feature_number == 0:
                        model += [
                            nn.Conv2d(in_channels=previous_feature, out_channels=channel_size, kernel_size=3,
                                      padding=1),
                            self.norm(channel_size), self.activation()]
                    elif feature_number == feature - 1:
                        model += [
                            nn.Conv2d(in_channels=previous_feature, out_channels=channel_size, kernel_size=3,
                                      padding=1),
                            self.norm(channel_size), self.activation(), pool(kernel_size=2, stride=2)]

                    else:
                        model += [
                            nn.Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=3, padding=1),
                            self.norm(channel_size), self.activation()]
                previous_feature = channel_size
            return model
        self.channel = channel
        self.max_channel = max_channel
        feature_list:List[int] = [2, 2, 3, 3, 3] #feature_list[i] := the number of times channel channel_size_list[i] is repeated
        channel_size_list: List[int] = [64, 128, 256, 512, 412]
        self.norm = norm
        self.activation = activation
        self.pool = nn.MaxPool2d
        model = create_model(feature_list= feature_list, channel_size_list=channel_size_list, input_channel=channel, pool =self.pool)
        self.pre_fusion = nn.Sequential(*model)



    def __call__(self, inputs: Tensor) -> Tensor:
        return super().__call__(inputs)

    def forward(self, inputs: Tensor) -> Tensor:
        if self.channel < self.max_channel:
            inputs = inputs.expand(inputs.size()[0], self.max_image_channel, inputs.size()[2], inputs.size()[3])
        return torch.flatten(self.pre_fusion(inputs), 1)


class Vgg16PostFusion(BaseModule):
    def __init__(self, name:str, vectorized_feature_length: int, number_of_classes: int,
                 activation: Callable[[], nn.Module] = lambda: nn.ReLU(inplace=True)):
        super().__init__(name=name)
        self.activation = activation
        feature_list: List[int] = [vectorized_feature_length, 4096, 4096, number_of_classes]
        model = self.create_model(feature_list=feature_list)
        self.post_fusion = nn.Sequential(*model)

    def create_model(self, feature_list: List[int]) -> List[nn.Module]:
        model = []
        for in_feature, out_feature in zip(feature_list, feature_list[1:]):
            model += [nn.Linear(in_feature, out_feature), self.activation()]
        return model

    def __call__(self, inputs: Tensor) -> Tensor:
        return super().__call__(inputs)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.post_fusion(inputs)
