from enum import Enum

from data.datasets.intel_scene_dataset import IntentSceneDataset





def data_set_parser(data_set_name:str, ):
    if data_set_name == 'intel':
        return IntentSceneDataset()
    else:
        print("no dataset")
        return None