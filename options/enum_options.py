import os
from typing import List, Tuple, Callable
from enum import Enum

from tap import Tap

from torchvision import transforms
from torchvision.transforms import CenterCrop

from data.datasets.intel_scene_dataset import IntentSceneDataset
from fusion_methods.full_connected_fusion import FullyConnectedFusion, FullyConnectedFusionFactory


class FusionOptions(Enum):
    FULLYCONNECTED = FullyConnectedFusionFactory(pretrain=False, freeze_features=False)



class TrainData(Enum):
    INTEL = IntentSceneDataset(dataset_directory = r"E:\famous_datasets\nirscene1", transform=transforms.Compose([CenterCrop(256),transforms.ToTensor(), lambda x: 2.0 * x - 1]), train_data_percent = .80, test_data = False)


class TrainingOptions(Tap):
    root = os.sep.join([os.getcwd(), 'outputs'])
    epoch_count = 1
    number_of_runs = 1
    batch_size = 10
    device = "cuda:0"


