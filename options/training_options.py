import os

from tap import Tap


class TrainingOptions(Tap):
    root = os.getcwd()
    epoch_count = 80
    number_of_runs = 1
    batch_size = 64
    device = "cuda:0"


