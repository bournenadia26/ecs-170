'''
CNN Model for MNIST - ECS 170 Stage 3
Input:  (N, 1, 28, 28) grayscale digit images
Output: 10-class digit classification (0-9)

Architecture (default):
  Conv1: 1->32, kernel=3, pad=1 -> ReLU -> MaxPool(2x2) => (32, 14, 14)
  Conv2: 32->64, kernel=3, pad=1 -> ReLU -> MaxPool(2x2) => (64, 7, 7)
  Dropout(0.25)
  FC1:  64*7*7=3136 -> 128 -> ReLU
  Dropout(0.5)
  FC2:  128 -> 10
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from base.base_method import Method


class Method_CNN_MNIST(Method, nn.Module):
    # default hyper-parameters
    max_epoch = 10
    learning_rate = 1e-3
    weight_decay = 1e-4

    def __init__(self, mName='CNN_MNIST', mDescription='CNN for MNIST digit classification',
                 n_classes=10, dropout=0.25):
        Method.__init__(self, mName=mName, mDescription=mDescription)
        nn.Module.__init__(self)

        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                          # -> (32, 14, 14)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                          # -> (64, 7, 7)

            nn.Dropout2d(dropout),
        )
        self.fc_block = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.fc_block(x)
        return x

    def train_model(self, train_loader, device):
        nn.Module.train(self)  # set training mode
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.learning_rate,
                                     weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        loss_history = []
        acc_history = []

        for epoch in range(self.max_epoch):
            total_loss = 0.0
            correct = 0
            total = 0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += imgs.size(0)

            scheduler.step()
            avg_loss = total_loss / total
            avg_acc  = correct / total
            loss_history.append(avg_loss)
            acc_history.append(avg_acc)
            print(f'  [MNIST] Epoch {epoch+1:02d}/{self.max_epoch} | Loss: {avg_loss:.4f} | Train Acc: {avg_acc:.4f}')

        return loss_history, acc_history

    def test_model(self, test_loader, device):
        nn.Module.eval(self)  # set eval mode
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(device)
                outputs = self(imgs)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.numpy().tolist())
        return all_preds, all_labels
