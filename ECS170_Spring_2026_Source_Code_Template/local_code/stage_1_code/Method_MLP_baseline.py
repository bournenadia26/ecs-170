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
        self.test_loss_history = []
        self.test_acc_history = []

    def _build_model(self, hidden_size=256, dropout_rate=0.0):
        self.fc_layer_1 = nn.Linear(784, hidden_size)
        self.activation_func_1 = nn.ReLU()
        self.dropout_1 = nn.Dropout(dropout_rate)

        self.fc_layer_2 = nn.Linear(hidden_size, hidden_size // 2)
        self.activation_func_2 = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout_rate)

        self.fc_layer_3 = nn.Linear(hidden_size // 2, 10)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation for toy dataset (4 features)'''
        h1 = self.activation_func_1(self.fc_layer_1(x))
        h2 = self.fc_layer_2(h1)
        y_pred = self.activation_func_2(h2)
        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        self.loss_history = []  # for storage so we can see
        self.acc_history = []
        self.test_loss_history = []
        self.test_acc_history = []

        # convert X into torch.tensor so pytorch algorithm can operate on it
        X_tensor = torch.FloatTensor(np.array(X))
        # convert y to torch.tensor as well
        y_true = torch.LongTensor(np.array(y))

        X_test_tensor = torch.FloatTensor(np.array(self.data['test']['X']))
        y_test_true = torch.LongTensor(np.array(self.data['test']['y']))

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        for epoch in range(self.max_epoch):  # you can do an early stop if self.max_epoch is too much...
            # get the output
            y_pred = self.forward(X_tensor)
            # calculate the training loss
            train_loss = loss_function(y_pred, y_true)

            # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            optimizer.zero_grad()
            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # do the error backpropagation to calculate the gradients
            train_loss.backward()
            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            optimizer.step()

            # Compute training accuracy and record loss and accuracy for plotting
            pred_labels = y_pred.max(1)[1]
            current_acc = np.mean((pred_labels == y_true).numpy())
            self.loss_history.append(train_loss.item())