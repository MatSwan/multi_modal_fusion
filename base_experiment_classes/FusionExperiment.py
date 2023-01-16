import os
from abc import abstractmethod
from typing import List
import torch.utils.data
from torch import Tensor
from torch.nn import ModuleList
from torch.optim import Optimizer

from base_experiment_classes.BaseExperiment import BaseExperiment
from models.base_module import BaseModule


class FusionExperiment(BaseExperiment):
    def __init__(self, root: str, experiment_name:str, train_data, test_data, feature_extractors: ModuleList,
                 pretrain_post_fusion_networks: ModuleList, epoch_count: int,
                 optimizers, pretrain_epoch, batch_size, number_of_runs:int = 1):
        self.train_data = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.test_data = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
        self._epoch_count = epoch_count
        self._optimizers: List[Optimizer] = optimizers
        self.feature_extractors = feature_extractors
        self.pretrain_post_fusion_networks = pretrain_post_fusion_networks
        self.pretrain_epoch = pretrain_epoch
        self.experiment_name = experiment_name
        self.number_of_runs = number_of_runs

        # self.pretrain_networks = pretrain  _networks
        super().__init__(root=root)

    def run(self, pretrain: bool, train_data=None, test_data=None, ):
        accuracy_list = []
        for i in range(self.number_of_runs):
            if train_data is not None: train_data = train_data
            if test_data is not None: test_data = test_data
            if pretrain:
                index_of_best_network = self.pretrain_on(train_data, test_data)
                self.load_pretrain_network(index_of_best_network)
            self.train_on(train_data)
            accuracy_list.append(self.do_test_on(test_data))
        self.save_accuracy(accuracy_list)

    def train_on(self, train_data):
        for epoch in range(self._epoch_count):
            for batch_data in train_data:
                self.do_batch_work(batch_data)

    def pretrain_on(self, train_data, test_data):
        for epoch in range(self.pretrain_epoch):
            for batch_data in train_data:
                self.do_batch_work(batch_data=batch_data, pretrain=True)
        return self.get_best_pretrained_network(test_data)

    def do_batch_work(self, batch_data, pretrain=False):
        for optimizer in self._optimizers: optimizer.zero_grad()
        if not pretrain:
            self.calculate_gradients(batch_data)
        else:
            self.pretrain_calculate_gradients(batch_data)

    def get_best_pretrained_network(self, test_data):
        correct_list = 0 * len(self.pretrain_post_fusion_networks)
        for batch_data in test_data:
            temp_correct_list = [
                self.feature_extractors[i](self.pretrain_post_fusion_networks[i](batch_data[0][i])) == batch_data[1] for
                i in range(len(self.pretrain_post_fusion_networks))]
            correct_list = [sum(x) for x in
                            zip(correct_list, temp_correct_list)]  # slow. A faster way is uses operator import add
        self.save(self.pretrain_post_fusion_networks)
        return correct_list.index(max(correct_list))


    def do_test_on(self, test_data) -> float:
        number_correct = 0
        for batch in test_data:
            input_data = batch[0]
            label = batch[1]
            classification_prediction = self.get_classification_prediction(input_data)
            if classification_prediction.item() == label.item():
                number_correct += 1
        return number_correct / len(test_data)

    def save_accuracy(self, accuracy_list):
        with open(os.path.join(self.root, self.experiment_name, 'results.txt')) as file:
            file.writelines(accuracy_list)


    @abstractmethod
    def get_classification_prediction(self, input_data) -> Tensor:
        pass

    @abstractmethod
    def calculate_gradients(self, batch_data):

        pass

    @abstractmethod
    def pretrain_calculate_gradients(self, batch_data):
        pass
    @abstractmethod
    def load_pretrain_network(self, index):
        pass