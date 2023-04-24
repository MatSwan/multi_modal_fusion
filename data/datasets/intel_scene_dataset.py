import os
from typing import List

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torchvision.transforms import CenterCrop


class IntentSceneDataset(Dataset):
    def __init__(self, dataset_directory: str = r"E:\famous_datasets\nirscene1",
                 transform=transforms.Compose([CenterCrop(256), transforms.ToTensor(), lambda x: 2.0 * x - 1]),
                 train_data_percent: float = .80, test_data: bool = False):
        self.name = 'intelscence'
        label = -1
        self.label_list = []
        self.dataset_directory = dataset_directory
        self.train_data_percent = train_data_percent
        self.transform = transform

        rgb_data = []
        nir_data = []
        for root, dirs, files in os.walk(dataset_directory):
            start_point = self.get_length(len(files), train_data_percent)
            if test_data:
                files = files[start_point:]  # Spliting test and train data
            else:
                files = files[:start_point]
            for name in files:
                if self.default_data_parser(name):
                    rgb_data.append(os.path.join(root, name))
                else:
                    nir_data.append(os.path.join(root, name))
            self.label_list += [label] * int(len(files) / 2)

            label += 1
        super().__init__()

        self.rgb_data = self.files_to_tensors(rgb_data)
        self.nir_data = self.files_to_tensors(nir_data)

        self.number_of_modalities = 2
        self.number_of_classes = 9
        self.image_channel: tuple = (3, 1)
        self.image_height: int = 256
        self.image_width: int = 256

    @staticmethod
    def default_data_parser(file: str):
        return 'rgb' in file

    @staticmethod
    def get_length(length: int, percent_of_train: float):
        start_point = round(length * percent_of_train)
        if start_point % 2 == 0:
            return start_point
        else:
            return start_point - 1

    def __len__(self):
        assert len(self.label_list) == len(self.rgb_data) and len(self.nir_data) == len(
            self.rgb_data), "The data has miss match lengths"
        return len(self.label_list)

    def files_to_tensors(self, file_list: List[str]):
        return [self.transform(Image.open(file)) for file in file_list]

    def __getitem__(self, index):
        rgb_data = self.rgb_data[index]
        nir_data = self.nir_data[index]
        label = self.label_list[index]
        input_tuple = tuple([rgb_data, nir_data])
        return tuple([input_tuple, label])

    def get_test_data(self):
        return IntentSceneDataset(dataset_directory=self.dataset_directory, transform=self.transform, test_data=True)


if __name__ == "__main__":
    j = IntentSceneDataset()
    t = j.get_test_data()
    2 + 2
