import os
from typing import List, Tuple, Callable

import torch
from torch import nn, Tensor
from torch.optim import Adam, SGD

from base_experiment_classes.FusionExperiment import FusionExperiment
from models.base_module import BaseModule
from models.util import ActivateGrad, set_requires_grad
from models.vgg_16_model.pre_post_vgg16 import Vgg16PreFusion, Vgg16PostFusion
from util.training_utill import NamedModuleList


class FusionVGG16(FusionExperiment):
    def __init__(self, root: str, experiment_name: str, train_data, test_data,
                 epoch_count: int, device: str,
                 pretrain_epoch, batch_size, number_of_modalities: int, number_of_classes: int,
                 image_channel: Tuple[int], image_height: int, image_width: int,
                 fusion: Callable[[int, int], nn.Module], norm_list: List[Callable[[int], nn.Module]] = None,
                 activation_list: List[Callable[[], nn.Module]] = None, number_of_runs: int = 1,
                 pretrain: bool = False, freeze_features: bool = False, prefusion_lr: float = 0.001, fusion_lr=.001,
                 postfusion_lr=0.001
                 ):
        '''

        :param root: Directory where oututs folder is
        :param experiment_name:  name used to create directory where results are stored (subdirectory of outputs folder)
        :param train_data: dataset to train the fusion vgg16 experiment with
        :param test_data: dataset to testthe fusion vgg16 experiment with
        :param epoch_count: number of epoch to train the model
        :param device: name of gpu to run the experiment on
        :param pretrain_epoch: number of epochs to train the pre fusion networks
        :param batch_size: batch size of train data
        :param number_of_modalities: number of modalities being fused togther
        :param number_of_classes:  number of classes in the training/testing dataset
        :param image_channel: tuple of ints where the ith int is the the input channel to the ith modality
        :param image_height: height of images containt in the training/testing dataset
        :param image_width: width of images contained in the training/testing dataset
        :param fusion: callable that returns the fusion module.
        :param norm_list: list that stores callables responsiable for creating the norm layers
        :param activation_list: list that stores callables responsiable for creating the activation layers
        :param number_of_runs: how many times should the experiment run
        :param pretrain: bool if True network will pretrain prefusion layers
        :param freeze_features: if True network will freeze the pretrained fusion layers
        :param prefusion_lr: learning rate for the prefusion portion of the network
        :param fusion_lr: learning rate for the fusion layer of the network
        :param postfusion_lr: learning rate for the postfusion layer of the network
        '''
        self.number_of_modalities = number_of_modalities

        # layers needed to make models
        norm_list = [nn.BatchNorm2d] * number_of_modalities if norm_list is None \
            else norm_list
        activation_list = [lambda: nn.ReLU(inplace=True)] * number_of_modalities if activation_list is None \
            else activation_list

        image_channel = image_channel  # Tuple that stores the number of channels found in the input for each feature extract (#todo fix this writing)

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
        fusion = fusion(number_of_modalities, vectorized_feature_number)
        # Construct post fusion layer
        postfusion = Vgg16PostFusion(name="vggpostfusion", vectorized_feature_length=vectorized_feature_number,
                                     number_of_classes=number_of_classes)
        prefusion_optimizers = [SGD(model.parameters(), lr=prefusion_lr, ) for model in
                                prefusion]
        fusion_optimizers = [SGD(fusion.parameters(), lr=fusion_lr, )]
        post_fusion_optimizers = [SGD(postfusion.parameters(), lr=postfusion_lr, )]
        optimizers = prefusion_optimizers + fusion_optimizers + post_fusion_optimizers

        if pretrain:
            # If we are pretraining we create M(number of modalities) distinct VGG networks
            pretrain_ends = nn.ModuleList([Vgg16PostFusion(name=f"pretrain_end{i}",
                                                           vectorized_feature_length=vectorized_feature_number,
                                                           number_of_classes=number_of_classes)
                                           for i in range(number_of_modalities)])
        else:
            pretrain_ends = None

        self.pretrain_directory = os.path.join(root, 'pretrained_networks')

        super().__init__(root=root, experiment_name=experiment_name, train_data=train_data, test_data=test_data,
                         prefusion=prefusion,
                         pretrain_ends=pretrain_ends, epoch_count=epoch_count, device=device,
                         optimizers=optimizers,
                         pretrain_epoch=pretrain_epoch, batch_size=batch_size, number_of_runs=number_of_runs, pretrain_directory=self.pretrain_directory)



        self.prefusion = NamedModuleList(nn.ModuleList([
            set_requires_grad(False, network) for network in prefusion
        ]), name='prefusion_modulelist')

        self.postfusion = set_requires_grad(False, postfusion)

        self.fusion = set_requires_grad(False, fusion)

        if pretrain:
            self.networks_to_pretrain = nn.ModuleList(self.prefusion + self.pretrain_ends)
            set_requires_grad(False, *self.pretrain_ends)

        # Loss function for the network
        self.loss = nn.CrossEntropyLoss()

        # networks to train Network_to_train holds that networks whose gradients are to be activate
        if freeze_features:
            self.networks_to_train = [self.postfusion]
        else:
            self.networks_to_train = [self.prefusion, self.fusion, self.postfusion]
        self.pretrain = pretrain
        self.to(device)

    def create_prefusion_network(self, norm: Callable[[int], nn.Module],
                                 activation: Callable[[], nn.Module], name: str, image_channel: int,
                                 max_channel) -> BaseModule:
        '''
        Method for crating prefusion data.
        norm: facotry for creating a norm layer
        activation: factory for creating an activation layer
        name: name of the prefusion network
        image_channel: number of channels for the given modality
        max_channel max number of channels for modalities
        '''
        return Vgg16PreFusion(name=name, channel=image_channel, max_channel=max_channel, norm=norm,
                              activation=activation)

    def calculate_gradients(self, batch_data: Tuple[Tuple[Tensor], Tensor]):
        with ActivateGrad(*self.networks_to_train):
            prefused_features: List[Tensor] = self.get_prefused_features(batch_data[0])
            fused_features = self.fuse_features(prefused_features)
            classification_guess = self.get_classification_guess(fused_features)
            loss = self.loss(classification_guess, batch_data[1])
            loss.backward()
            return loss.item()

    def get_prefused_features(self, input_data: Tuple[Tensor]) -> List[Tensor]:
        return [self.prefusion[index](inputs) for index, inputs in enumerate(input_data)]

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

    def load_pretrain_features(self,):
        "takes an index of the best pretrained model and loads it."
        temp_network_list: List[BaseModule] = []
        for i in range(self.number_of_modalities):
            temp_network = self.create_prefusion_network(name=self.prefusion[i].get_name(), norm=self.norm_list[i],
                                                         activation=self.activation_list[i],
                                                         max_channel=max(self.image_channel),
                                                         image_channel=self.image_channel[i])
            temp_network_list.append(temp_network)
        self.prefusion = nn.ModuleList(temp_network_list)
        self.prefusion.state_dict = torch.load(
            os.sep.join([self.pretrain_directory, 'prefusion_modulelist.pth']))

    @staticmethod
    def get_vectoried_feature_number(number_of_channels, height, width, prefusion_layer):
        return prefusion_layer(torch.ones(1, number_of_channels, height, width)).size()[-1]

    def get_test_prediction(self, input_data) -> Tensor:
        prefused_features: List[Tensor] = self.get_prefused_features(input_data)
        fused_features = self.fuse_features(prefused_features)
        classification_guess = torch.argmax(torch.nn.functional.softmax(\
                                            self.get_classification_guess(fused_features)))
        return classification_guess

    def forward(self, input_data):
        prefused_features: List[Tensor] = self.get_prefused_features(input_data)
        fused_features = self.fuse_features(prefused_features)
        classification_guess = self.get_classification_guess(fused_features)
        return classification_guess

