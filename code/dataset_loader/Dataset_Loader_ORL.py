'''
Dataset Loader for ORL Face Dataset - ECS 170 Stage 3
ORL: 360 train / 40 test, 112x92x3 grayscale (RGB identical), 40 classes (person IDs 1-40)
We use only the R channel -> shape (N, 1, 112, 92), labels shifted to 0-indexed (0-39)
'''

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from base.base_dataset import Dataset_Loader


class ORL_Dataset(Dataset):
    def __init__(self, images, labels):
        # images: (N, 112, 92, 3) or (N, 112, 92) -> use 1 channel
        self.images = images
        self.labels = labels  # 0-indexed

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        # Take R channel if 3-channel, then add channel dim
        if img.ndim == 3:
            img = img[:, :, 0]   # (112, 92)
        img = img[np.newaxis, :, :].astype(np.float32) / 255.0  # (1, 112, 92)
        return torch.tensor(img, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


class Dataset_Loader_ORL(Dataset_Loader):
    def __init__(self, seed=None, dName='ORL'):
        super().__init__(seed=seed, dName=dName)
        self.data_path = None
        self.batch_size = 16  # small dataset, small batch

    def load(self):
        assert self.data_path is not None, 'Set data_path before calling load()'
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)

        train_images = np.array([inst['image'] for inst in data['train']])
        train_labels = np.array([inst['label'] - 1 for inst in data['train']])  # shift 1-40 -> 0-39
        test_images  = np.array([inst['image'] for inst in data['test']])
        test_labels  = np.array([inst['label'] - 1 for inst in data['test']])

        train_dataset = ORL_Dataset(train_images, train_labels)
        test_dataset  = ORL_Dataset(test_images, test_labels)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                  num_workers=0, pin_memory=False)
        test_loader  = DataLoader(test_dataset,  batch_size=self.batch_size, shuffle=False,
                                  num_workers=0, pin_memory=False)

        print(f'[ORL] Train: {len(train_dataset)} | Test: {len(test_dataset)}')
        return train_loader, test_loader
