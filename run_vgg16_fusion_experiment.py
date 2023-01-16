'''script used to run vgg16 experiment. Note befor running please go enum options and set the directory path for each dataset
links to download each dataset can be found in readme.txt'''
import os

from options.enum_options import TrainData, FusionOptions

DataSetOptions = TrainData
FusionOptions = FusionOptions
#options for running the experiment
root = os.getcwd()
number_of_runs = 1
train_set = DataSetOptions.INTEL.value
test_set=train_set.get_test_data
number_of_modalities = train_set.number_of_modalities
number_of_classes = train_set.number_of_classes
train_options = get_train_options

