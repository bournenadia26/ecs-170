'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    # ADDED BC NO NEED TO SPLIT WITH THIS
    dataset_source_file_name_train = None
    dataset_source_file_name_test = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data...')
        """
        ORIGINAL CODE PRESERVED JUST IN CASE!!
        X = []
        y = []
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'r')
        for line in f:
            line = line.strip('\n')
            elements = [int(i) for i in line.split(' ')] # <-- ORIGINAL
            X.append(elements[:-1])
            y.append(elements[-1])
        f.close()
        return {'X': X, 'y': y}
        """

        # prof already gave us seperated train.csv and test.csv. Original code literally will not work??
        # new version to account for pre-seperated files
        X_train, y_train = [], []
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name_train, 'r')
        for line in f:
            line = line.strip('\n')
            elements = [int(i) for i in line.strip().split(',')] # comma seperated instead of space bc he gave us csv's
            y_train.append(elements[0]) # label first as specified in README for stage2
            X_train.append(elements[1:])
        f.close()

        X_test, y_test = [], []
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name_test, 'r')
        for line in f:
            line = line.strip('\n')
            elements = [int(i) for i in line.strip().split(',')]
            y_test.append(elements[0])
            X_test.append(elements[1:])
        f.close()

        return {
            'train': {'X': X_train, 'y': y_train},
            'test': {'X': X_test, 'y': y_test}
        }