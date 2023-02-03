import os
from abc import abstractmethod
from typing import List
import torch.utils.data
from torch import Tensor
from torch.nn import ModuleList
from torch.optim import Optimizer

from base_experiment_classes.BaseExperiment import BaseExperiment



class FusionExperiment(BaseExperiment):
    def __init__(self, root: str, experiment_name:str, train_data, test_data, prefusion: ModuleList,
                 pretrain_ends: ModuleList, epoch_count: int,
                 optimizers, pretrain_epoch, batch_size, number_of_runs:int = 1):
        '''

        :param root: str stores root directory for saveing of modules
        :param experiment_name: str name of experiment
        :param train_data: data loader that holds train data. See  #todo insert file with dataloader documentation
        :param test_data:  data loader that holds test data.
        :param prefusion: nn.ModuleList that holds all network responsible for the prefusion calculations
        :param pretrain_ends: ModuleList storing the postfusion pairs for prefusion feature extractor. Used in Pretraining
        :param epoch_count: Number of epochs in traininig
        :param optimizers: List of Optimizers used in training
        :param pretrain_epoch: Number of epochs in pretraining
        :param batch_size: Batch size used in training
        :param number_of_runs: Number of experimental trials
        '''
        super().__init__(root=root)
        self.train_data = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.test_data = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
        self._epoch_count = epoch_count
        self._optimizers: List[Optimizer] = optimizers
        self.prefusion = prefusion
        self.pretrain_ends = pretrain_ends
        self.pretrain_epoch = pretrain_epoch
        self.experiment_name = experiment_name
        self.number_of_runs = number_of_runs

        # self.pretrain_networks = pretrain  _networks

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
        return self.get_best_pretrained_network(train_data)

    def do_batch_work(self, batch_data, pretrain=False):
        for optimizer in self._optimizers: optimizer.zero_grad()
        if not pretrain:
            self.calculate_gradients(batch_data)
        else:
            self.pretrain_calculate_gradients(batch_data)

    def get_best_pretrained_network(self, test_data):
        correct_list = 0 * len(self.pretrain_ends)
        for batch_data in test_data:
            temp_correct_list = [
                self.prefusion[i](self.pretrain_ends[i](batch_data[0][i])) == batch_data[1] for
                i in range(len(self.pretrain_ends))]
            correct_list = [sum(x) for x in
                            zip(correct_list, temp_correct_list)]  # slow. A faster way is uses operator import add
        self.save(self.pretrain_ends)
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