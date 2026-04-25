from pathlib import Path
import sys

# Ensure imports work no matter where the script is launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_code.stage_1_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP, tune_mlp
from local_code.stage_1_code.Result_Saver import Result_Saver
#from local_code.stage_2_code.Setting_KFold_CV import Setting_KFold_CV
from local_code.stage_1_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from local_code.stage_1_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch
import matplotlib.pyplot as plt

#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_folder = PROJECT_ROOT / "data" / "stage_2_data"

    data_obj = Dataset_Loader('MNIST', '')
    data_obj.dataset_source_folder_path = str(data_folder) + '/'
    data_obj.dataset_source_file_name_train = 'train.csv'
    data_obj.dataset_source_file_name_test = 'test.csv'

    method_obj = Method_MLP('multi-layer perceptron', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_1_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    #setting_obj = Setting_KFold_CV('k fold cross validation', '') # <-- instructions state no CV this time
    setting_obj = Setting_Train_Test_Split('train test split', '')
    #setting_obj = Setting_Tra
    # in_Test_Split('train test split', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # load data once for tuning
    loaded_data = data_obj.load()

    # autotune
    print('----- BEGIN -----')
    tune_result = tune_mlp(loaded_data)
    print('----- END -----')
    # ------------------------------------------------------

    # plotting
    if tune_result is not None and 'best_history' in tune_result and tune_result['best_history'] is not None:
        loss_history = tune_result['best_history']['loss_history']
        acc_history = tune_result['best_history']['acc_history']
        test_loss_history = tune_result['best_history']['test_loss_history']
        test_acc_history = tune_result['best_history']['test_acc_history']

        plot_folder = PROJECT_ROOT / "result" / "stage_2_result"
        plot_folder.mkdir(parents=True, exist_ok=True)

        step = 5

        sampled_epochs = list(range(1, len(loss_history) + 1, step))
        sampled_train_loss = loss_history[::step]
        sampled_test_loss = test_loss_history[::step]
        sampled_train_acc = acc_history[::step]
        sampled_test_acc = test_acc_history[::step]

        if sampled_epochs[-1] != len(loss_history):
            sampled_epochs.append(len(loss_history))
            sampled_train_loss.append(loss_history[-1])
            sampled_test_loss.append(test_loss_history[-1])
            sampled_train_acc.append(acc_history[-1])
            sampled_test_acc.append(test_acc_history[-1])

        best_config = tune_result['best_config']
        best_accuracy = tune_result['best_accuracy']

        # create figure with two subplots
        plt.figure(figsize=(12, 5))

        # plot the loss curve
        plt.subplot(1, 2, 1)
        plt.plot(sampled_epochs, sampled_train_loss, label='Train Loss')
        plt.plot(sampled_epochs, sampled_test_loss, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve (Sampled Every 5 Epochs)')
        plt.grid(True)
        plt.legend()

        # mark the final loss point
        last_epoch = len(loss_history)
        last_train_loss = loss_history[-1]
        last_test_loss = test_loss_history[-1]
        plt.scatter(last_epoch, last_train_loss)
        plt.scatter(last_epoch, last_test_loss)
        plt.text(last_epoch, last_train_loss, f'{last_train_loss:.4f}', fontsize=9, ha='left', va='bottom')
        plt.text(last_epoch, last_test_loss, f'{last_test_loss:.4f}', fontsize=9, ha='left', va='top')

        # plot the accuracy curve
        plt.subplot(1, 2, 2)
        plt.plot(sampled_epochs, sampled_train_acc, label='Train Accuracy')
        plt.plot(sampled_epochs, sampled_test_acc, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curve (Sampled Every 5 Epochs)')
        plt.grid(True)
        plt.legend()

        # mark the final accuracy point
        last_train_acc = acc_history[-1]
        last_test_acc = test_acc_history[-1]
        plt.scatter(last_epoch, last_train_acc)
        plt.scatter(last_epoch, last_test_acc)
        plt.text(last_epoch, last_train_acc, f'{last_train_acc:.4f}', fontsize=9, ha='left', va='bottom')
        plt.text(last_epoch, last_test_acc, f'{last_test_acc:.4f}', fontsize=9, ha='left', va='top')

        # overall title
        plt.suptitle('MLP Convergence Curves', fontsize=14)

        # add the best config and test accuracy into the figure
        config_text = (
            f"Best Config: hidden_size={best_config['hidden_size']}, "
            f"lr={best_config['lr']}, epochs={best_config['epochs']}, "
            f"dropout={best_config.get('dropout', 0.0)}    "
            f"Final Test Accuracy={best_accuracy:.4f}"
        )
        plt.figtext(0.5, 0.90, config_text, ha='center', fontsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.88])

        # save and display the figure
        plt.savefig(plot_folder / 'MLP_training_progress.png')
        plt.show()

        print('Plots saved to:', plot_folder)
    else:
        print('No training history returned from tune_mlp()')