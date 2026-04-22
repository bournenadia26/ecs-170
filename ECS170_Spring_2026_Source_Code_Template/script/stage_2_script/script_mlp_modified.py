<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 50f8de9 (pathname error fix)
from pathlib import Path
import sys

# Ensure imports work no matter where the script is launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

<<<<<<< HEAD
from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_1_code.Result_Saver import Result_Saver
#from local_code.stage_1_code.Setting_KFold_CV import Setting_KFold_CV
=======
from local_code.stage_1_code.Dataset_Loader import Dataset_Loader
from local_code.stage_1_code.Method_MLP import Method_MLP, tune_mlp
from local_code.stage_1_code.Result_Saver import Result_Saver
from local_code.stage_1_code.Setting_KFold_CV import Setting_KFold_CV
>>>>>>> 24bef01 (copied script_mlp into stage_2, tweaked for given dataset, called autotuning method)
from local_code.stage_1_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from local_code.stage_1_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
from pathlib import Path
import sys

# Ensure imports work no matter where the script is launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_1_code.Result_Saver import Result_Saver
=======
from local_code.stage_1_code.Dataset_Loader import Dataset_Loader
from local_code.stage_1_code.Method_MLP import Method_MLP, tune_mlp
from local_code.stage_1_code.Result_Saver import Result_Saver
>>>>>>> 50f8de9 (pathname error fix)
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
<<<<<<< HEAD
<<<<<<< HEAD
    data_folder = PROJECT_ROOT / "data" / "stage_2_data"

    data_obj = Dataset_Loader('MNIST', '')
    data_obj.dataset_source_folder_path = str(data_folder) + '/'
=======
    script_dir = Path(__file__).resolve().parent

    data_obj = Dataset_Loader('MNIST', '')
    data_obj.dataset_source_folder_path = str(script_dir) + '/'
>>>>>>> 50f8de9 (pathname error fix)
=======
    data_folder = PROJECT_ROOT / "data" / "stage_2_data"

    data_obj = Dataset_Loader('MNIST', '')
    data_obj.dataset_source_folder_path = str(data_folder) + '/'
>>>>>>> 0ec6ea8 (added plotting for loss and accuracy curves for the best config model after tuning)
    data_obj.dataset_source_file_name_train = 'train.csv'
    data_obj.dataset_source_file_name_test = 'test.csv'

    method_obj = Method_MLP('multi-layer perceptron', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    #setting_obj = Setting_KFold_CV('k fold cross validation', '') # <-- instructions state no CV this time
    setting_obj = Setting_Train_Test_Split('train test split', '')
    #setting_obj = Setting_Tra
    # in_Test_Split('train test split', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # load data
    loaded_data = data_obj.load()

<<<<<<< HEAD
    # Example: train and evaluate MLP
    print('----- BEGIN TRAINING -----')
    X_train = np.array(loaded_data['train']['X'], dtype=np.float32)
    y_train = np.array(loaded_data['train']['y'], dtype=np.int64)
    X_test = np.array(loaded_data['test']['X'], dtype=np.float32)
    y_test = np.array(loaded_data['test']['y'], dtype=np.int64)

    model = method_obj
    optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    train_loss_history = []
    test_loss_history = []
    train_acc_history = []
    test_acc_history = []
    X_train_tensor = torch.from_numpy(X_train)
    y_train_tensor = torch.from_numpy(y_train)
    X_test_tensor = torch.from_numpy(X_test)
    y_test_tensor = torch.from_numpy(y_test)
    for epoch in range(model.max_epoch):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        # Training accuracy
        _, train_pred = torch.max(outputs, 1)
        train_acc = (train_pred == y_train_tensor).float().mean().item()
        train_loss_history.append(loss.item())
        train_acc_history.append(train_acc)
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_loss = criterion(test_outputs, y_test_tensor).item()
            _, test_pred = torch.max(test_outputs, 1)
            test_acc = (test_pred == y_test_tensor).float().mean().item()
            test_loss_history.append(test_loss)
            test_acc_history.append(test_acc)
        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}/{model.max_epoch}, Train Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    print('----- END TRAINING -----')

    # Save results (evaluate on test set again to ensure predictions are up to date)
    model.eval()
    with torch.no_grad():
        final_test_outputs = model(X_test_tensor)
        _, final_test_pred = torch.max(final_test_outputs, 1)
        result_obj.data = final_test_pred.numpy().tolist()
    result_obj.fold_count = 0  # Ensure filename is valid
    result_obj.save()

    # Plotting
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(train_loss_history, label='Train Loss')
    axs[0].plot(test_loss_history, label='Test Loss')
    axs[0].set_title('Loss Curve')
    axs[0].legend()
    axs[1].plot(train_acc_history, label='Train Accuracy')
    axs[1].plot(test_acc_history, label='Test Accuracy')
    axs[1].set_title('Accuracy Curve')
    axs[1].legend()
    plt.suptitle('MLP Training Progress')
    # Save the plot as a PNG file automatically
    plot_save_path = PROJECT_ROOT / "result" / "stage_2_result" / "MLP_training_progress.png"
    plt.savefig(str(plot_save_path))
    print(f"Plot saved to {plot_save_path}")
    plt.show()
    loaded_data = data_obj.load()
=======
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
>>>>>>> 0ec6ea8 (added plotting for loss and accuracy curves for the best config model after tuning)
