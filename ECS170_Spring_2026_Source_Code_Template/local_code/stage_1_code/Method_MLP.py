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
        # This file is intentionally left as a stub after extracting both ablation and baseline versions.
        # Please use Method_MLP_ablation.py or Method_MLP_baseline.py for your experiments.
    def tune_mlp(data):
        # Feel free to modify and see what works better!!
        hidden_sizes = [512]  # neurons per layer
        learning_rates = [1e-3]  # learning rates
        epoch_counts = [500]  # epochs
        dropout_rates = [0.0, 0.3]

        # for plotting test
        # hidden_sizes = [256]
        # learning_rates = [1e-3]
        # epoch_counts = [50]
        # dropout_rates = [0.0]

        best_accuracy = 0
        best_config = None
        best_history = None

        for hidden_size in hidden_sizes:  # for every combination of every hyperparam:
            for lr in learning_rates:
                for epochs in epoch_counts:
                    for dropout_rate in dropout_rates:
                        print(
                            f'\n--- Trying hidden={hidden_size}, lr={lr}, epochs={epochs}, dropout={dropout_rate} ---')

                        model = Method_MLP('mlp', '')
                        model._build_model(hidden_size=hidden_size, dropout_rate=dropout_rate)
                        model.learning_rate = lr
                        model.max_epoch = epochs
                        model.data = data

                        result = model.run()

                        pred_y = result['pred_y'].numpy()
                        true_y = np.array(result['true_y'])

                        # metrics
                        accuracy = np.mean(pred_y == true_y)
                        precision = precision_score(true_y, pred_y, average='weighted')
                        recall = recall_score(true_y, pred_y, average='weighted')
                        f1 = f1_score(true_y, pred_y, average='weighted')

                        print("----- FINAL STATS -----")
                        print(
                            f'Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}')

                        if accuracy > best_accuracy:  # best model is judged purely by accuracy
                            best_accuracy = accuracy
                            best_config = {
                                'hidden_size': hidden_size,
                                'lr': lr,
                                'epochs': epochs,
                                'dropout': dropout_rate
                            }
                            best_history = {
                                'loss_history': result['loss_history'],
                                'acc_history': result['acc_history'],
                                'test_loss_history': result['test_loss_history'],
                                'test_acc_history': result['test_acc_history']
                            }

        print('\n----- BEST CONFIG:', best_config, '| Accuracy:', best_accuracy, '-----')
        return {
            'best_config': best_config,
            'best_accuracy': best_accuracy,
            'best_history': best_history
        }


def tune_mlp(data):
    """Module-level wrapper so callers can import tune_mlp directly."""
    return Method_MLP.tune_mlp(data)
