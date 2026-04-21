'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_1_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

class Method_MLP(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 500
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        #self.fc_layer_1 = nn.Linear(4, 4)
        # check here for nn.ReLU doc: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        #self.activation_func_1 = nn.ReLU()
        #self.fc_layer_2 = nn.Linear(4, 2)
        # check here for nn.Softmax doc: https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
        #self.activation_func_2 = nn.Softmax(dim=1)

        """Fundamental restructure bc we're not working with tiny toy dataset anymore"""
        # For stage 1 toy dataset: 4 input features, 1 output (binary classification)
        self.fc_layer_1 = nn.Linear(4, 8)
        self.activation_func_1 = nn.ReLU()
        self.fc_layer_2 = nn.Linear(8, 2)
        self.activation_func_2 = nn.Softmax(dim=1)

        # loss and accuracy lists for plotting
        self.loss_history = []
        self.acc_history = []

    def _build_model(self, hidden_size=256): # 2 hidden layers. feel free to tweak to more. compresses to 256 neurons.
        self.fc_layer_1 = nn.Linear(784, hidden_size) #<- stage 2 number of features
        self.activation_func_1 = nn.ReLU()
        self.fc_layer_2 = nn.Linear(hidden_size, hidden_size)
        self.activation_func_2 = nn.ReLU()
        self.fc_layer_3 = nn.Linear(hidden_size, 10)
