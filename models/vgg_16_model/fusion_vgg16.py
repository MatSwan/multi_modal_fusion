import os
from collections import Callable
from typing import List, Tuple

import torch
from torch import nn, Tensor
from torch.optim import Adam

from base_experiment_classes.FusionExperiment import FusionExperiment
from models.base_module import BaseModule
from models.util import ActivateGrad, set_requires_grad
from models.vgg_16_model.pre_post_vgg16 import Vgg16PreFusion, Vgg16PostFusion


class FusionVGG16(FusionExperiment):
    def __init__(self, root: str, experiment_name:str, train_data, test_data,
                 epoch_count: int,
                  pretrain_epoch, batch_size, number_of_modalities: int, vectorized_feature_number: int, number_of_classes: int,
                 fusion: Callable[[int], nn.Module], norm_list: List[Callable[[int], nn.Module]] = None,
                 activation_list: List[Callable[[], nn.Module]] = None, number_of_runs:int = None,
                 pretrain: bool = False, freeze_features: bool = False, prefusion_lr:float =0.001, fusion_lr=.001, postfusion_lr=0.001
                 ):
        self.number_of_modalities = number_of_modalities

        #layers needed to make models
        self.norm_list = [nn.BatchNorm2d] * number_of_modalities if norm_list is None \
                         else norm_list
        self.activation_list = [lambda: nn.ReLU(inplace=True)] * number_of_modalities if activation_list is None \
                                else activation_list
        #Construct prefusion layers
        self.prefusion = nn.ModuleList([
            self.create_prefusion_network(name=f'vgg_prefusion_modality_network{i}', norm=self.norm_list[i], activation=self.activation_list[i])
            for i in range(number_of_modalities)
                                        ])
        #Construct fusion layer
        self.fusion = fusion(number_of_modalities)

        #Construct post fusion layer
        self.postfusion = Vgg16PostFusion(vectorized_feature_length=vectorized_feature_number,
                                          number_of_classes=number_of_classes)
        prefusion_optimizers = [Adam(model.parameters(), lr=prefusion_lr,  betas=(0.9, 0.999)) for model in self.prefusion]
        fusion_optimizers = [Adam(self.fusion.parameters(), lr=fusion_lr,  betas=(0.9, 0.999))]
        post_fusion_optimizers = [Adam(self.postfusion.parameters(), lr=postfusion_lr, betas=(0.9, 0.999))]
        optimizers = prefusion_optimizers + fusion_optimizers + post_fusion_optimizers

        if pretrain:
            #If we are pretraining we create M(number of modalities) distinct VGG networks
            self.pretrain_ends = nn.ModuleList([Vgg16PostFusion(name=f"pretrain_end{i}", vectorized_feature_length=vectorized_feature_number, number_of_classes=number_of_classes)
                                                for i in range(number_of_modalities)])
            self.networks_to_pretrain = nn.ModuleList(self.prefusion + self.pretrain_ends)
        else: self.pretrain_ends = None


        #turn off grads. We turn grads on via our ActivateGrad with statment durning training
        set_requires_grad(False, *self.prefusion)
        set_requires_grad(False, self.postfusion)
        if pretrain: set_requires_grad(False, *self.pretrain_ends)

        #set loss function
        self.loss = nn.CrossEntropyLoss()

        #networks to train
        if freeze_features: self.networks_to_train = [self.postfusion]
        else: self.networks_to_train = nn.ModuleList(self.prefusion + [self.postfusion])

        self.final_outout_directory = os.path.join(root, experiment_name, 'final_results')
        self.pretrain_directory = os.path.join(root, experiment_name, 'pretrained_networks')

        super().__init__(root = root, experiment_name=experiment_name, train_data=train_data, test_data=test_data, feature_extractors=self.prefusion,
                         pretrain_post_fusion_networks=self.pretrain_ends,epoch_count=epoch_count, optimizers=optimizers,
                         pretrain_epoch=pretrain_epoch,batch_size=batch_size,number_of_runs=number_of_runs)

    def create_prefusion_network(self, norm: Callable[[int], nn.Module],
                                 activation: Callable[[], nn.Module], name:str) ->BaseModule:
        return Vgg16PreFusion(name=name, norm=norm, activation=activation)

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

    def get_classification_guess(self, fused_features: Tensor)->Tensor:
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
        prefused_features: List[Tensor] = self.get_prefused_features(input_data)
        fused_features = self.fuse_features(prefused_features)
        classification_guess = torch.nn.functional.softmax(self.get_classification_guess(fused_features))
        return classification_guess

    def load_pretrain_network(self, index):
        temp_network_list: List[BaseModule] = []
        for i in range(self.number_of_modalities):
            temp_network = self.create_prefusion_network(name=self.prefusion[i].get_name(), norm=self.norm_list[i], activation=self.activation_list[i])
            if i == index:
                temp_network.state_dict = torch.load(os.path.join(self.pretrain_directory, self.prefusion[i].get_name()))
            temp_network_list.append(temp_network)
        self.prefusion = nn.ModuleList(temp_network_list)