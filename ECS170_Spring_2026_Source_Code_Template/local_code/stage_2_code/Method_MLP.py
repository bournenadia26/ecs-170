# Copy of Method_MLP.py for stage 2. Update input/output sizes and metrics as needed.

from local_code.base_class.method import method
from torch import nn
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

class Method_MLP(method, nn.Module):
    data = None
    max_epoch = 500
    learning_rate = 1e-3

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # For MNIST: 784 input features, 10 output classes
        self.fc1 = nn.Linear(784, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, 10)
        self.loss_history = []

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

    # Add train, test, and run methods as in stage 1, but adapt for multiclass
