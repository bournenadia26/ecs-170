'''
CNN Model for CIFAR-10 - ECS 170 Stage 3
Input:  (N, 3, 32, 32) colored object images
Output: 10-class object classification (0-9)

Architecture: ResNet-18 adapted for CIFAR-10 (32x32 images)
  - Stem: Conv(3->64, k=3, pad=1) + BN + ReLU  [no MaxPool — images too small]
  - Layer1: 2x BasicBlock(64->64,  stride=1)    => (64,  32, 32)
  - Layer2: 2x BasicBlock(64->128, stride=2)    => (128, 16, 16)
  - Layer3: 2x BasicBlock(128->256,stride=2)    => (256,  8,  8)
  - Layer4: 2x BasicBlock(256->512,stride=2)    => (512,  4,  4)
  - AdaptiveAvgPool(1x1) -> Dropout -> FC(512->10)

Training:
  - SGD + Nesterov momentum=0.9, weight_decay=5e-4
  - OneCycleLR (max_lr=0.1, pct_start=0.3)
  - Label smoothing=0.1
  Expected test accuracy: ~92-94%
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from base.base_method import Method


class _BasicBlock(nn.Module):
    """Standard residual block with optional projection shortcut."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class Method_CNN_CIFAR(Method, nn.Module):
    max_epoch     = 60
    learning_rate = 1e-3        # Adam LR
    weight_decay  = 1e-4

    def __init__(self, mName='CNN_CIFAR', mDescription='ResNet-18 for CIFAR-10 classification',
                 n_classes=10, dropout=0.0):
        Method.__init__(self, mName=mName, mDescription=mDescription)
        nn.Module.__init__(self)
        self.dropout_p = dropout

        # Stem: 3x3 conv, no maxpool (CIFAR images are only 32x32)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(64,  64,  n_blocks=2, stride=1)
        self.layer2 = self._make_layer(64,  128, n_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, n_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, n_blocks=2, stride=2)

        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(512, n_classes)

        # Kaiming init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels, out_channels, n_blocks, stride):
        layers = [_BasicBlock(in_channels, out_channels, stride)]
        for _ in range(1, n_blocks):
            layers.append(_BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def train_model(self, train_loader, device):
        nn.Module.train(self)
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        # Cosine annealing per epoch — MPS-compatible
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epoch, eta_min=1e-5
        )

        loss_history = []
        acc_history  = []

        for epoch in range(self.max_epoch):
            total_loss = 0.0
            correct = 0
            total   = 0
            for batch_idx, (imgs, labels) in enumerate(train_loader):
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += imgs.size(0)

                if batch_idx == 0:
                    print(f'  [CIFAR-10] Epoch {epoch+1:02d} started, first batch loss={loss.item():.4f}', flush=True)
                elif batch_idx % 50 == 0:
                    print(f'    batch {batch_idx}/{len(train_loader)} loss={loss.item():.4f}', flush=True)

            scheduler.step()
            avg_loss = total_loss / total
            avg_acc  = correct / total
            loss_history.append(avg_loss)
            acc_history.append(avg_acc)
            print(f'  [CIFAR-10] Epoch {epoch+1:02d}/{self.max_epoch} | Loss: {avg_loss:.4f} | Train Acc: {avg_acc:.4f}', flush=True)

        return loss_history, acc_history

    def test_model(self, test_loader, device):
        nn.Module.eval(self)
        all_preds  = []
        all_labels = []
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(device)
                outputs = self(imgs)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.numpy().tolist())
        return all_preds, all_labels
