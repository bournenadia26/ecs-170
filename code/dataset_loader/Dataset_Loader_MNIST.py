'''
Dataset Loader for MNIST - ECS 170 Stage 3
MNIST: 60000 train / 10000 test, 28x28 grayscale images, 10 classes (digits 0-9)
Returns tensors of shape (N, 1, 28, 28) and labels (N,)
'''

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from base.base_dataset import Dataset_Loader


class MNIST_Dataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images   # numpy array (N, 28, 28) or (N, 1, 28, 28)
        self.labels = labels   # numpy array (N,)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        # shape (28, 28) -> (1, 28, 28), normalize to [0,1]
        if img.ndim == 2:
            img = img[np.newaxis, :, :]
        img = img.astype(np.float32) / 255.0
        return torch.tensor(img, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


class Dataset_Loader_MNIST(Dataset_Loader):
    def __init__(self, seed=None, dName='MNIST'):
        super().__init__(seed=seed, dName=dName)
        self.data_path = None
        self.batch_size = 64

    def load(self):
        assert self.data_path is not None, 'Set data_path before calling load()'
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)

        train_images = np.array([inst['image'] for inst in data['train']])
        train_labels = np.array([inst['label'] for inst in data['train']])
        test_images  = np.array([inst['image'] for inst in data['test']])
        test_labels  = np.array([inst['label'] for inst in data['test']])

        train_dataset = MNIST_Dataset(train_images, train_labels)
        test_dataset  = MNIST_Dataset(test_images, test_labels)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                  num_workers=0, pin_memory=False)
        test_loader  = DataLoader(test_dataset,  batch_size=self.batch_size, shuffle=False,
                                  num_workers=0, pin_memory=False)

        print(f'[MNIST] Train: {len(train_dataset)} | Test: {len(test_dataset)}')
        return train_loader, test_loader
