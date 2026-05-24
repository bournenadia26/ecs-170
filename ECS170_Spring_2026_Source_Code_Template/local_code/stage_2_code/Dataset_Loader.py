# Dataset loader for stage 2 (e.g., MNIST CSVs)
from local_code.base_class.dataset import dataset
import numpy as np

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name_train = None
    dataset_source_file_name_test = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        X_train, y_train = [], []
        with open(self.dataset_source_folder_path + self.dataset_source_file_name_train, 'r') as f:
            for line in f:
                elements = [int(i) for i in line.strip().split(',')]
                y_train.append(elements[0])
                X_train.append(elements[1:])
        X_test, y_test = [], []
        with open(self.dataset_source_folder_path + self.dataset_source_file_name_test, 'r') as f:
            for line in f:
                elements = [int(i) for i in line.strip().split(',')]
                y_test.append(elements[0])
                X_test.append(elements[1:])
        return {
            'train': {'X': X_train, 'y': y_train},
            'test': {'X': X_test, 'y': y_test}
        }
