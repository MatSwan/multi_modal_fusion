import os
from collections.abc import Callable
from typing import List, Tuple

import torch
from torch import nn, Tensor
from torch.optim import Adam

from base_experiment_classes.FusionExperiment import FusionExperiment
from models.base_module import BaseModule
from models.util import ActivateGrad, set_requires_grad
from models.vgg_16_model.pre_post_vgg16 import Vgg16PreFusion, Vgg16PostFusion


class FusionVGG16(FusionExperiment):
    def __init__(self, root: str, experiment_name: str, train_data, test_data,
                 epoch_count: int, device: str,
                 pretrain_epoch, batch_size, number_of_modalities: int, number_of_classes: int,
                 image_channel: Tuple[int], image_height: int, image_width: int,
                 fusion: Callable[[int], nn.Module], norm_list: List[Callable[[int], nn.Module]] = None,
                 activation_list: List[Callable[[], nn.Module]] = None, number_of_runs: int = None,
                 pretrain: bool = False, freeze_features: bool = False, prefusion_lr: float = 0.001, fusion_lr=.001,
                 postfusion_lr=0.001
                 ):
        self.number_of_modalities = number_of_modalities

        # layers needed to make models
        norm_list = [nn.BatchNorm2d] * number_of_modalities if norm_list is None \
            else norm_list
        activation_list = [lambda: nn.ReLU(inplace=True)] * number_of_modalities if activation_list is None \
            else activation_list

        image_channel = image_channel #Tuple that stores the number of channels found in the input for each feature extract (#todo fix this writting)

        # Construct prefusion layers
        prefusion = nn.ModuleList([
            self.create_prefusion_network(name=f'vgg_prefusion_modality_network{i}', norm=norm_list[i],
                                          activation=activation_list[i],
                                          image_channel=image_channel[i], max_channel=max(image_channel))
            for i in range(number_of_modalities)
        ])

        # Construct fusion layer
        vectorized_feature_number = self.get_vectoried_feature_number(number_of_channels=max(image_channel),
                                                                      height=image_height,
                                                                      width=image_width,
                                                                      prefusion_layer=prefusion[0])
        fusion = fusion(number_of_modalities)
        # Construct post fusion layer
        postfusion = Vgg16PostFusion(name="vggpostfusion", vectorized_feature_length=vectorized_feature_number,
                                          number_of_classes=number_of_classes)
        prefusion_optimizers = [Adam(model.parameters(), lr=prefusion_lr, betas=(0.9, 0.999)) for model in
                                prefusion]
        fusion_optimizers = [Adam(fusion.parameters(), lr=fusion_lr, betas=(0.9, 0.999))]
        post_fusion_optimizers = [Adam(self.postfusion.parameters(), lr=postfusion_lr, betas=(0.9, 0.999))]
        optimizers = prefusion_optimizers + fusion_optimizers + post_fusion_optimizers

        if pretrain:
            # If we are pretraining we create M(number of modalities) distinct VGG networks
            pretrain_ends = nn.ModuleList([Vgg16PostFusion(name=f"pretrain_end{i}",
                                                                vectorized_feature_length=vectorized_feature_number,
                                                                number_of_classes=number_of_classes)
                                                for i in range(number_of_modalities)])
        else:
            pretrain_ends = None


        self.final_outout_directory = os.path.join(root, experiment_name, 'final_results')
        self.pretrain_directory = os.path.join(root, experiment_name, 'pretrained_networks')

        super().__init__(root=root, experiment_name=experiment_name, train_data=train_data, test_data=test_data,
                         prefusion=prefusion,
                         pretrain_ends=pretrain_ends, epoch_count=epoch_count,
                         optimizers=optimizers,
                         pretrain_epoch=pretrain_epoch, batch_size=batch_size, number_of_runs=number_of_runs)

        self.to(device)

        #TODO FIX ALL OF THIS STUFF

        # turn off grads. We turn grads on via our ActivateGrad with statment durning training


        self.prefusion = nn.ModuleList([
            set_requires_grad(False, network) for network in prefusion
        ])

        self.postfusion = set_requires_grad(False, postfusion)

        self.fusion = set_requires_grad(False, fusion(number_of_modalities))

        if pretrain:
            self.networks_to_pretrain = nn.ModuleList(self.prefusion + self.pretrain_ends)
            set_requires_grad(False, *self.pretrain_ends)

        #Loss function for the network
        self.loss = nn.CrossEntropyLoss()

        # networks to train Network_to_train holds that networks whose gradients are to be activate
        if freeze_features:
            self.networks_to_train = [self.postfusion]
        else:
            self.networks_to_train = nn.ModuleList(self.prefusion + [self.postfusion])




    def create_prefusion_network(self, norm: Callable[[int], nn.Module],
                                 activation: Callable[[], nn.Module], name: str, image_channel: int,
                                 max_channel) -> BaseModule:
        '''Method for crating prefusion data.
        norm: facotry for creating a norm layer
        activation: factory for creating an activation layer
        name: name of the prefusion network
        image_channel: number of channels for the given modality
        max_channel max number of channels for modalities
        '''
        return Vgg16PreFusion(name=name, channel=image_channel, max_channel=max_channel, norm=norm,
                              activation=activation)

    def calculate_gradients(self, batch_data: Tuple[Tuple[Tensor], Tensor]):
        with ActivateGrad(self.networks_to_train):
            prefused_features: List[Tensor] = self.get_prefused_features(batch_data[0])
            fused_features = self.fuse_features(prefused_features)
            classification_guess = self.get_classification_guess(fused_features)
            loss = self.loss(classification_guess, batch_data[1])
            loss.backward()

    def get_prefused_features(self, input_data: Tuple[Tensor]) -> List[Tensor]:
        return [self.prefusion(inputs) for inputs in input_data]

    def fuse_features(self, prefused_features: List[Tensor]) -> Tensor:
        return self.fusion(prefused_features)

    def get_classification_guess(self, fused_features: Tensor) -> Tensor:
        return self.postfusion(fused_features)

    def pretrain_calculate_gradients(self, batch_data: Tuple[Tuple[Tensor], Tensor]) -> None:
        with ActivateGrad(self.networks_to_pretrain):
            model_features: List[Tensor] = self.get_prefused_features(batch_data[0])
            classification_guess = self.get_pretrain_classifications(model_features)
            self.multi_network_loss(pretrain_classifications=classification_guess, label=batch_data[1])

    def get_pretrain_classifications(self, model_features):
        return [self.pretrain_ends[i](model_feature) for i, model_feature in enumerate(model_features)]

    def multi_network_loss(self, pretrain_classifications, label):
        for pretrain_classification in pretrain_classifications:
            self.loss(pretrain_classification, label).backward()

    def get_output(self, input_data):
        ""
        prefused_features: List[Tensor] = self.get_prefused_features(input_data)
        fused_features = self.fuse_features(prefused_features)
        classification_guess = torch.nn.functional.softmax(self.get_classification_guess(fused_features))
        return classification_guess

    def load_pretrain_network(self, index):
        "takes an index of the best pretrained model and loads it."
        temp_network_list: List[BaseModule] = []
        for i in range(self.number_of_modalities):
            temp_network = self.create_prefusion_network(name=self.prefusion[i].get_name(), norm=self.norm_list[i],
                                                         activation=self.activation_list[i], max_channel=max(self.image_channel),image_channel=self.image_channel[i])
            if i == index:
                temp_network.state_dict = torch.load(
                    os.path.join(self.pretrain_directory, self.prefusion[i].get_name()))
            temp_network_list.append(temp_network)
        self.prefusion = nn.ModuleList(temp_network_list)
    @staticmethod
    def get_vectoried_feature_number(number_of_channels, height, width, prefusion_layer):
        return prefusion_layer(torch.ones(1, number_of_channels, height, width)).size().item()
