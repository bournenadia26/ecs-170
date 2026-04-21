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
<<<<<<< HEAD
<<<<<<< HEAD
from sklearn.metrics import precision_score, recall_score, f1_score
=======

>>>>>>> 5788125 (initial commit)
=======
from sklearn.metrics import precision_score, recall_score, f1_score
>>>>>>> bc33ae7 (import scikit learn in Method_MLP oops sorry)

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
<<<<<<< HEAD
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

=======
        self.fc_layer_1 = nn.Linear(4, 4)
        # check here for nn.ReLU doc: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        #self.activation_func_1 = nn.ReLU()
        #self.fc_layer_2 = nn.Linear(4, 2)
        # check here for nn.Softmax doc: https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
        self.activation_func_2 = nn.Softmax(dim=1)

>>>>>>> 5788125 (initial commit)
    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
<<<<<<< HEAD
        '''Forward propagation for toy dataset (4 features)'''
        h1 = self.activation_func_1(self.fc_layer_1(x))
        h2 = self.fc_layer_2(h1)
        y_pred = self.activation_func_2(h2)
=======
        '''Forward propagation'''
        # hidden layer embeddings
        h1 = self.activation_func_1(self.fc_layer_1(x))
        h2 = self.activation_func_2(self.fc_layer_2(h1))
        # outout layer result
        # self.fc_layer_2(h) will be a nx2 tensor
        # n (denotes the input instance number): 0th dimension; 2 (denotes the class number): 1st dimension
        # we do softmax along dim=1 to get the normalized classification probability distributions for each instance
        y_pred = self.activation_func_2(self.fc_layer_2(h))
>>>>>>> 5788125 (initial commit)
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

<<<<<<< HEAD
        self.loss_history = [] # for storage so we can see

=======
>>>>>>> 5788125 (initial commit)
        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            y_pred = self.forward(torch.FloatTensor(np.array(X)))
            # convert y to torch.tensor as well
            y_true = torch.LongTensor(np.array(y))
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

<<<<<<< HEAD
            # Compute training accuracy and record loss and accuracy for plotting
            pred_labels = y_pred.max(1)[1]
            current_acc = np.mean((pred_labels == y_true).numpy())
            self.loss_history.append(train_loss.item())
            self.acc_history.append(current_acc)

            if (epoch + 1) % 10 == 0: # adjusted because epoch number starts at 0
                accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                print('Epoch:', epoch + 1, 'Accuracy:', current_acc, 'Loss:', train_loss.item()) # a little uninformative for epochs=100
=======
            if epoch%100 == 0:
                accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
<<<<<<< HEAD
                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())
>>>>>>> 5788125 (initial commit)
=======
                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item()) # a little uninformative for epochs=100
>>>>>>> 6848cdd (tune_mlp() import error resolved)
    
    def test(self, X):
        # do the testing, and result the result
        y_pred = self.forward(torch.FloatTensor(np.array(X)))
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]
    
    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
<<<<<<< HEAD
        return {'pred_y': pred_y,
                'true_y': self.data['test']['y'],
                'loss_history': self.loss_history,
                'acc_history': self.acc_history
        }

    """Method to auto-tune hyperparameters. Tries every combination of listed hyperparams and prints the results."""
    @staticmethod
    def tune_mlp(data):
        # Feel free to modify and see what works better!!
        hidden_sizes = [128, 256, 512] # neurons per layer
        learning_rates = [1e-3, 1e-4] # learning rates
        epoch_counts = [100, 300, 500] # epochs

<<<<<<< HEAD
        # for plotting test
        # hidden_sizes = [256]
        # learning_rates = [1e-3]
        # epoch_counts = [50]

=======
>>>>>>> 6848cdd (tune_mlp() import error resolved)
        best_accuracy = 0
        best_config = None
        best_history = None

        for hidden_size in hidden_sizes: # for every combination of every hyperparam:
            for lr in learning_rates:
                for epochs in epoch_counts:
                    print(f'\n--- Trying hidden={hidden_size}, lr={lr}, epochs={epochs} ---')

                    model = Method_MLP('mlp', '')
                    model._build_model(hidden_size=hidden_size)
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
                    print(f'Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}')

                    if accuracy > best_accuracy: # best model is judged purely by accuracy
                        best_accuracy = accuracy
                        best_config = {'hidden_size': hidden_size, 'lr': lr, 'epochs': epochs}
                        best_history = {
                            'loss_history': result['loss_history'],
                            'acc_history': result['acc_history']
                        }

<<<<<<< HEAD
<<<<<<< HEAD
        print('\n----- BEST CONFIG:', best_config, '| Accuracy:', best_accuracy, '-----')
        return {
            'best_config': best_config,
            'best_accuracy': best_accuracy,
            'best_history': best_history
        }
=======
        print('\n*** Best config:', best_config, '| Accuracy:', best_accuracy, '***')
>>>>>>> 6848cdd (tune_mlp() import error resolved)
=======
        print('\n----- BEST CONFIG:', best_config, '| Accuracy:', best_accuracy, '-----')
>>>>>>> ae1001f (improved clarity for reading output print statements)


def tune_mlp(data):
    """Module-level wrapper so callers can import tune_mlp directly."""
    return Method_MLP.tune_mlp(data)
<<<<<<< HEAD
=======
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
            
>>>>>>> 5788125 (initial commit)
=======
>>>>>>> 6848cdd (tune_mlp() import error resolved)
