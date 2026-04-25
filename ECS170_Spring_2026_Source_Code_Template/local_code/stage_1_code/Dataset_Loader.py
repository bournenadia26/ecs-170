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
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> d627638 (modified Dataset_Loader.py to work with csv's and pre split datasets)

    # ADDED BC NO NEED TO SPLIT WITH THIS
    dataset_source_file_name_train = None
    dataset_source_file_name_test = None
<<<<<<< HEAD
=======
>>>>>>> 5788125 (initial commit)
=======
>>>>>>> d627638 (modified Dataset_Loader.py to work with csv's and pre split datasets)
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data...')
<<<<<<< HEAD
<<<<<<< HEAD
        """
        ORIGINAL CODE PRESERVED JUST IN CASE!!
=======
>>>>>>> 5788125 (initial commit)
=======
        """
        ORIGINAL CODE PRESERVED JUST IN CASE!!
>>>>>>> d627638 (modified Dataset_Loader.py to work with csv's and pre split datasets)
        X = []
        y = []
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'r')
        for line in f:
            line = line.strip('\n')
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 74723fc (Baseline commit before ablation studies (original MLP architecture, results, and scripts))
            elements = [int(i) for i in line.strip().split()]
            X.append(elements[:-1])
            y.append(elements[-1])
        f.close()
        return {'X': X, 'y': y}
        """

        # prof already gave us seperated train.csv and test.csv. Original code literally will not work??
        # new version to account for pre-seperated files
        # Stage 2: If both train and test file names are set, load as train/test split
        if self.dataset_source_file_name_train and self.dataset_source_file_name_test:
<<<<<<< HEAD
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
                # For pre-split datasets
                dataset_source_file_name_train = None
                dataset_source_file_name_test = None

                def __init__(self, dName=None, dDescription=None):
                    super().__init__(dName, dDescription)

                def load(self):
                    print('loading data...')
                    # If both train and test file names are set, load as train/test split
                    if self.dataset_source_file_name_train and self.dataset_source_file_name_test:
                        X_train, y_train = [], []
                        with open(self.dataset_source_folder_path + self.dataset_source_file_name_train, 'r') as f:
                            for line in f:
                                line = line.strip('\n')
                                elements = [int(i) for i in line.strip().split()]
                                y_train.append(elements[0])
                                X_train.append(elements[1:])
                        X_test, y_test = [], []
                        with open(self.dataset_source_folder_path + self.dataset_source_file_name_test, 'r') as f:
                            for line in f:
                                line = line.strip('\n')
                                elements = [int(i) for i in line.strip().split()]
                                y_test.append(elements[0])
                                X_test.append(elements[1:])
                        return {
                            'train': {'X': X_train, 'y': y_train},
                            'test': {'X': X_test, 'y': y_test}
                        }
                    # fallback to original logic for non-split datasets
                    elif self.dataset_source_file_name:
                        X = []
                        y = []
                        with open(self.dataset_source_folder_path + self.dataset_source_file_name, 'r') as f:
                            for line in f:
                                line = line.strip('\n')
                                elements = [int(i) for i in line.strip().split()]
                                X.append(elements[:-1])
                                y.append(elements[-1])
                        return {'X': X, 'y': y}
                    else:
                        raise ValueError('No dataset source file(s) specified.')
            y_train.append(elements[0]) # label first as specified in README for stage2
=======
            X_train, y_train = [], []
            f = open(self.dataset_source_folder_path + self.dataset_source_file_name_train, 'r')
            for line in f:
                line = line.strip('\n')
                elements = [int(i) for i in line.strip().split()]
                y_train.append(elements[0])
                X_train.append(elements[1:])
            f.close()

            X_test, y_test = [], []
            f = open(self.dataset_source_folder_path + self.dataset_source_file_name_test, 'r')
            for line in f:
                line = line.strip('\n')
                elements = [int(i) for i in line.strip().split()]
                y_test.append(elements[0])
                X_test.append(elements[1:])
            f.close()

            return {
                'train': {'X': X_train, 'y': y_train},
                'test': {'X': X_test, 'y': y_test}
            }
        # Stage 1: Otherwise, load as single file with X/y
        else:
            X = []
            y = []
            f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'r')
            for line in f:
                line = line.strip('\n')
                elements = [int(i) for i in line.strip().split()]
                X.append(elements[:-1])
                y.append(elements[-1])
            f.close()
            return {'X': X, 'y': y}
>>>>>>> 74723fc (Baseline commit before ablation studies (original MLP architecture, results, and scripts))
