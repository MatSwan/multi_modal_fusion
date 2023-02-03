from enum import Enum
from torchvision import transforms
from data.datasets.intel_scene_dataset import IntentSceneDataset
from fusion_methods.full_connected_fusion import FullyConnectedFusion, FullyConnectedFusionFactory


class FusionOptions(Enum):
    FULLYCONNECTED = FullyConnectedFusionFactory(pretrain=False, freeze_features=False)



class TrainData(Enum):
    INTEL = IntentSceneDataset(dataset_directory = r"E:\famous_datasets\nirscene1", transform=transforms.Compose([transforms.ToTensor(), lambda x: 2.0 * x - 1]), train_data_percent = .80, test_data = False)