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


class Method_CNN(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 50
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3
    # mini-batch sizes for training and evaluation
    batch_size = 64
    eval_batch_size = 256

    # it defines the CNN model architecture:
    # convolution layers, pooling layers, and fully connected layers.
    # input is expected as (N, C, H, W) — batch, channels, height, width
    def __init__(self, mName, mDescription, in_channels=1, img_size=28, num_classes=10):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # store these so _build_model and forward can reference them
        self.in_channels = in_channels
        self.img_size = img_size
        self.num_classes = num_classes

        self._build_model(
            num_filters=32,
            fc_hidden_size=256,
            dropout_rate=0.0
        )

        # loss and accuracy lists for plotting
        self.loss_history = []
        self.acc_history = []
        self.test_loss_history = []
        self.test_acc_history = []

    # ---------------------------------------------------------------------------
    # Architecture builder
    # ---------------------------------------------------------------------------
    def _build_model(self, num_filters=32, fc_hidden_size=256, dropout_rate=0.0):
        """
        Build the CNN layers.

        Convolutional block 1: conv -> ReLU -> max-pool
        Convolutional block 2: conv -> ReLU -> max-pool
        Fully connected block:  flatten -> fc -> ReLU -> dropout -> fc (output)

        After two max-pool(2x2) layers the spatial size is img_size // 4.
        The flattened size fed into the first FC layer is therefore:
            (num_filters * 2) * (img_size // 4) * (img_size // 4)
        """
        # --- Conv block 1 ---
        # check here for nn.Conv2d doc: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        self.conv_layer_1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=num_filters,       # number of filters / feature maps
            kernel_size=3,                  # 3x3 sliding window
            stride=1,
            padding=1                       # 'same' padding keeps H and W unchanged
        )
        self.activation_func_1 = nn.ReLU()
        # check here for nn.MaxPool2d doc: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # halves H and W

        # --- Conv block 2 ---
        self.conv_layer_2 = nn.Conv2d(
            in_channels=num_filters,
            out_channels=num_filters * 2,   # double the filters in the second block
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.activation_func_2 = nn.ReLU()
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Fully connected block ---
        # spatial size after two pool layers: img_size // 4
        conv_out_size = (self.img_size // 4) * (self.img_size // 4) * (num_filters * 2)

        self.fc_layer_1 = nn.Linear(conv_out_size, fc_hidden_size)
        self.activation_func_3 = nn.ReLU()
        self.dropout_1 = nn.Dropout(dropout_rate)

        self.fc_layer_2 = nn.Linear(fc_hidden_size, self.num_classes)

        # save for possible rebuild later
        self._num_filters = num_filters
        self._fc_hidden_size = fc_hidden_size
        self._dropout_rate = dropout_rate

    # ---------------------------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------------------------
    def forward(self, x):
        '''Forward propagation'''
        # x shape: (N, C, H, W)

        # conv block 1
        h = self.activation_func_1(self.conv_layer_1(x))
        h = self.pool_1(h)

        # conv block 2
        h = self.activation_func_2(self.conv_layer_2(h))
        h = self.pool_2(h)

        # flatten: (N, filters*2, H', W') -> (N, filters*2 * H' * W')
        h = h.view(h.size(0), -1)

        # fully connected block
        h = self.activation_func_3(self.fc_layer_1(h))
        h = self.dropout_1(h)

        y_pred = self.fc_layer_2(h)
        return y_pred

    # ---------------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------------
    def fit(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        self.loss_history = []
        self.acc_history = []
        self.test_loss_history = []
        self.test_acc_history = []

        # X is expected to already be shaped (N, C, H, W) — supplied by Dataset_Loader
        X_tensor = torch.FloatTensor(np.array(X))
        y_true = torch.LongTensor(np.array(y))

        X_test_tensor = torch.FloatTensor(np.array(self.data['test']['X']))
        y_test_true = torch.LongTensor(np.array(self.data['test']['y']))

        train_dataset = torch.utils.data.TensorDataset(X_tensor, y_true)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_true)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.eval_batch_size, shuffle=False
        )

        for epoch in range(self.max_epoch):
            # put model in training mode (enables dropout etc.)
            self.train()
            epoch_train_loss_sum = 0.0
            epoch_train_correct = 0
            epoch_train_total = 0

            for X_batch, y_batch in train_loader:
                y_pred = self.forward(X_batch)
                train_loss = loss_function(y_pred, y_batch)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                batch_size = y_batch.size(0)
                pred_labels = y_pred.max(1)[1]
                epoch_train_loss_sum += train_loss.item() * batch_size
                epoch_train_correct += (pred_labels == y_batch).sum().item()
                epoch_train_total += batch_size

            avg_train_loss = epoch_train_loss_sum / epoch_train_total
            current_acc = epoch_train_correct / epoch_train_total
            self.loss_history.append(avg_train_loss)
            self.acc_history.append(current_acc)

            # test metrics (no gradient needed)
            # put model in eval mode (disables dropout etc.)
            self.eval()
            with torch.no_grad():
                epoch_test_loss_sum = 0.0
                epoch_test_correct = 0
                epoch_test_total = 0

                for X_batch, y_batch in test_loader:
                    test_pred = self.forward(X_batch)
                    test_loss = loss_function(test_pred, y_batch)
                    test_labels = test_pred.max(1)[1]

                    batch_size = y_batch.size(0)
                    epoch_test_loss_sum += test_loss.item() * batch_size
                    epoch_test_correct += (test_labels == y_batch).sum().item()
                    epoch_test_total += batch_size

                avg_test_loss = epoch_test_loss_sum / epoch_test_total
                test_acc = epoch_test_correct / epoch_test_total

            self.test_loss_history.append(avg_test_loss)
            self.test_acc_history.append(test_acc)

            if (epoch + 1) % 10 == 0:
                print(
                    'Epoch:', epoch + 1,
                    'Train Accuracy:', round(current_acc, 4),
                    'Train Loss:', round(avg_train_loss, 4),
                    'Test Accuracy:', round(test_acc, 4),
                    'Test Loss:', round(avg_test_loss, 4)
                )

    # ---------------------------------------------------------------------------
    # Testing
    # ---------------------------------------------------------------------------
    def test(self, X):
        self.eval()
        X_tensor = torch.FloatTensor(np.array(X))
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_tensor),
            batch_size=self.eval_batch_size,
            shuffle=False
        )

        pred_chunks = []
        with torch.no_grad():
            for (X_batch,) in test_loader:
                y_pred = self.forward(X_batch)
                pred_chunks.append(y_pred.max(1)[1])
        return torch.cat(pred_chunks, dim=0)

    # ---------------------------------------------------------------------------
    # Run (called by the training scripts)
    # ---------------------------------------------------------------------------
    def run(self):
        print('method running...')
        print('--start training...')
        self.fit(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {
            'pred_y': pred_y,
            'true_y': self.data['test']['y'],
            'loss_history': self.loss_history,
            'acc_history': self.acc_history,
            'test_loss_history': self.test_loss_history,
            'test_acc_history': self.test_acc_history
        }

    # ---------------------------------------------------------------------------
    # Hyperparameter tuning helper (mirrors tune_mlp pattern)
    # ---------------------------------------------------------------------------
    @staticmethod
    def tune_cnn(data, in_channels=1, img_size=28, num_classes=10):
        """
        Tries every combination of listed hyperparams and prints the results.
        Mirrors the style of Method_MLP.tune_mlp.
        Feel free to modify and see what works better!
        """
        num_filters_list  = [32]
        fc_hidden_sizes   = [256]
        learning_rates    = [1e-3]
        epoch_counts      = [50]
        dropout_rates     = [0.0]

        best_accuracy = 0
        best_config   = None
        best_history  = None

        for num_filters in num_filters_list:
            for fc_hidden_size in fc_hidden_sizes:
                for lr in learning_rates:
                    for epochs in epoch_counts:
                        for dropout_rate in dropout_rates:
                            print(
                                f'\n--- Trying filters={num_filters}, fc_hidden={fc_hidden_size}, '
                                f'lr={lr}, epochs={epochs}, dropout={dropout_rate} ---'
                            )

                            model = Method_CNN('cnn', '',
                                               in_channels=in_channels,
                                               img_size=img_size,
                                               num_classes=num_classes)
                            model._build_model(
                                num_filters=num_filters,
                                fc_hidden_size=fc_hidden_size,
                                dropout_rate=dropout_rate
                            )
                            model.learning_rate = lr
                            model.max_epoch = epochs
                            model.data = data

                            result = model.run()

                            pred_y = result['pred_y'].numpy()
                            true_y = np.array(result['true_y'])

                            accuracy  = np.mean(pred_y == true_y)
                            precision = precision_score(true_y, pred_y, average='weighted', zero_division=0)
                            recall    = recall_score(true_y, pred_y, average='weighted', zero_division=0)
                            f1        = f1_score(true_y, pred_y, average='weighted', zero_division=0)

                            print("----- FINAL STATS -----")
                            print(
                                f'Accuracy: {accuracy:.4f} | Precision: {precision:.4f} '
                                f'| Recall: {recall:.4f} | F1: {f1:.4f}'
                            )

                            if accuracy > best_accuracy:
                                best_accuracy = accuracy
                                best_config = {
                                    'num_filters': num_filters,
                                    'fc_hidden_size': fc_hidden_size,
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


def tune_cnn(data, in_channels=1, img_size=28, num_classes=10):
    """Module-level wrapper so callers can import tune_cnn directly."""
    return Method_CNN.tune_cnn(data, in_channels=in_channels,
                               img_size=img_size, num_classes=num_classes)