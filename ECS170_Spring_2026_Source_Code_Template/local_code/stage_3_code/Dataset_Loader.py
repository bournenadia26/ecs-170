import pickle
from PIL import Image
import numpy as np

class Dataset_Loader:
    def __init__(self, dataset_name, dataset_path):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path

    def load(self):
        with open(self.dataset_path, 'rb') as f:
            raw_data = pickle.load(f)

        X_train, y_train = self.convert_split(raw_data['train'])
        X_test, y_test = self.convert_split(raw_data['test'])

        return {
            'train': {
                'X': X_train,
                'y': y_train
            },
            'test': {
                'X': X_test,
                'y': y_test
            }
        }

    def convert_split(self, split_data):
        X = []
        y = []

        for instance in split_data:
            image = np.array(instance['image'])
            label = int(instance['label'])

            if self.dataset_name == 'MNIST':
                image = image.astype(np.float32) / 255.0
                image = image.reshape(1, 28, 28)

            elif self.dataset_name == 'CIFAR':
                image = image.astype(np.float32) / 255.0
                image = np.transpose(image, (2, 0, 1))

            elif self.dataset_name == 'ORL':
                image = image[:, :, 0]

                image = Image.fromarray(image.astype(np.uint8))
                image = image.resize((92, 92))
                image = np.array(image, dtype=np.float32) / 255.0

                image = image.reshape(1, 92, 92)

                label = label - 1

            else:
                raise ValueError('Unknown dataset name: ' + self.dataset_name)

            X.append(image)
            y.append(label)

        return np.array(X), np.array(y)