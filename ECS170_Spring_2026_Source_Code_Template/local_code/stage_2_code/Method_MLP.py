<<<<<<< HEAD
# This file is intentionally left as a stub after extracting both ablation and baseline versions.
# Please use Method_MLP_ablation.py or Method_MLP_baseline.py for your experiments.
            self.acc_history.append(current_acc)

            with torch.no_grad():
                test_pred = self.forward(X_test_tensor)
                test_loss = loss_function(test_pred, y_test_true)
                test_labels = test_pred.max(1)[1]
                test_acc = np.mean((test_labels == y_test_true).numpy())

            self.test_loss_history.append(test_loss.item())
            self.test_acc_history.append(test_acc)

            if (epoch + 1) % 10 == 0:  # adjusted because epoch number starts at 0
                accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                print(
                    'Epoch:', epoch + 1,
                    'Train Accuracy:', current_acc,
                    'Train Loss:', train_loss.item(),
                    'Test Accuracy:', test_acc,
                    'Test Loss:', test_loss.item()
                )

    def test(self, X):
        # do the testing, and result the result
        with torch.no_grad():
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
        return {'pred_y': pred_y,
                'true_y': self.data['test']['y'],
                'loss_history': self.loss_history,
                'acc_history': self.acc_history,
                'test_loss_history': self.test_loss_history,
                'test_acc_history': self.test_acc_history
                }

    """Method to auto-tune hyperparameters. Tries every combination of listed hyperparams and prints the results."""

    @staticmethod
    def tune_mlp(data):
        # Feel free to modify and see what works better!!
        hidden_sizes = [512]  # neurons per layer
        learning_rates = [1e-3]  # learning rates
        epoch_counts = [300]  # epochs
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
=======
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
        self.fc1 = nn.Linear(784, 128)
        self.tanh1 = nn.Tanh()
        self.dropout1 = nn.Dropout(0.0)
        self.fc2 = nn.Linear(128, 10)
        self.loss_history = []

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

    # Add train, test, and run methods as in stage 1, but adapt for multiclass
>>>>>>> e0998a4 (Baseline commit before ablation studies (original MLP architecture, results, and scripts))
