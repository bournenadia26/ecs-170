from pathlib import Path
import sys

# Ensure imports work no matter where the script is launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_code.stage_1_code.Dataset_Loader import Dataset_Loader
from local_code.stage_1_code.Method_MLP import Method_MLP, tune_mlp
from local_code.stage_1_code.Result_Saver import Result_Saver
#from local_code.stage_1_code.Setting_KFold_CV import Setting_KFold_CV
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

        plot_folder = PROJECT_ROOT / "result" / "stage_2_result"
        plot_folder.mkdir(parents=True, exist_ok=True)

        step = 5

        sampled_loss_epochs = list(range(1, len(loss_history) + 1, step))
        sampled_loss_values = loss_history[::step]
        if sampled_loss_epochs[-1] != len(loss_history):
            sampled_loss_epochs.append(len(loss_history))
            sampled_loss_values.append(loss_history[-1])

        sampled_acc_epochs = list(range(1, len(acc_history) + 1, step))
        sampled_acc_values = acc_history[::step]
        if sampled_acc_epochs[-1] != len(acc_history):
            sampled_acc_epochs.append(len(acc_history))
            sampled_acc_values.append(acc_history[-1])

        # create figure with two subplots
        plt.figure(figsize=(12, 5))

        # plot the loss curve
        plt.subplot(1, 2, 1)
        plt.plot(sampled_loss_epochs, sampled_loss_values)
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title('Loss Curve (Sampled Every 5 Epochs)')
        plt.grid(True)

        # mark the final loss point
        last_epoch_loss = len(loss_history)
        last_loss = loss_history[-1]
        plt.scatter(last_epoch_loss, last_loss)
        plt.text(last_epoch_loss, last_loss, f'{last_loss:.4f}', fontsize=9, ha='left', va='bottom')

        # plot the accuracy curve
        plt.subplot(1, 2, 2)
        plt.plot(sampled_acc_epochs, sampled_acc_values)
        plt.xlabel('Epoch')
        plt.ylabel('Training Accuracy')
        plt.title('Accuracy Curve (Sampled Every 5 Epochs)')
        plt.grid(True)

        # mark the final accuracy point
        last_epoch_acc = len(acc_history)
        last_acc = acc_history[-1]
        plt.scatter(last_epoch_acc, last_acc)
        plt.text(last_epoch_acc, last_acc, f'{last_acc:.4f}', fontsize=9, ha='left', va='bottom')

        # overall title
        plt.suptitle('MLP Convergence Curves during Training', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # save and display the figure
        plt.savefig(plot_folder / 'mlp_curve.png')
        plt.show()

        print('Plots saved to:', plot_folder)
    else:
        print('No training history returned from tune_mlp()')