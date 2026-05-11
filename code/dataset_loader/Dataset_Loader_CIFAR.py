'''
Dataset Loader for CIFAR-10 - ECS 170 Stage 3
CIFAR-10: 50000 train / 10000 test, 32x32x3 colored images, 10 classes (objects 0-9)
Training: RandomCrop + RandomHorizontalFlip + channel normalization (CIFAR-10 stats)
Test: channel normalization only
'''

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from base.base_dataset import Dataset_Loader

# CIFAR-10 channel statistics (pre-computed over training set)
_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD  = (0.2470, 0.2435, 0.2616)


class CIFAR_Dataset(Dataset):
    def __init__(self, images, labels, transform=None):
        # images: (N, 32, 32, 3) uint8 HWC format
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]  # (32, 32, 3) uint8
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.tensor(
                np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0,
                dtype=torch.float32
            )
        return img, torch.tensor(self.labels[idx], dtype=torch.long)


class Dataset_Loader_CIFAR(Dataset_Loader):
    def __init__(self, seed=None, dName='CIFAR-10'):
        super().__init__(seed=seed, dName=dName)
        self.data_path = None
        self.batch_size = 512

    def load(self):
        assert self.data_path is not None, 'Set data_path before calling load()'
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)

        train_images = np.array([inst['image'] for inst in data['train']])
        train_labels = np.array([inst['label'] for inst in data['train']])
        test_images  = np.array([inst['image'] for inst in data['test']])
        test_labels  = np.array([inst['label'] for inst in data['test']])

        # Augmentation for training; only normalization for test
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=_CIFAR_MEAN, std=_CIFAR_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
        ])
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=_CIFAR_MEAN, std=_CIFAR_STD),
        ])

        train_dataset = CIFAR_Dataset(train_images, train_labels, transform=train_transform)
        test_dataset  = CIFAR_Dataset(test_images,  test_labels,  transform=test_transform)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                  num_workers=0, pin_memory=False)
        test_loader  = DataLoader(test_dataset,  batch_size=self.batch_size, shuffle=False,
                                  num_workers=0, pin_memory=False)

        print(f'[CIFAR-10] Train: {len(train_dataset)} | Test: {len(test_dataset)}')
        return train_loader, test_loader
