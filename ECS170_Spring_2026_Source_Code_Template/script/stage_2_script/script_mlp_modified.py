from pathlib import Path
import sys

# Ensure imports work no matter where the script is launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

<<<<<<< HEAD
from local_code.stage_1_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP, tune_mlp
=======
from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
>>>>>>> e0998a4 (Baseline commit before ablation studies (original MLP architecture, results, and scripts))
from local_code.stage_1_code.Result_Saver import Result_Saver
#from local_code.stage_2_code.Setting_KFold_CV import Setting_KFold_CV
from local_code.stage_1_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from local_code.stage_1_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch
import matplotlib.pyplot as plt

#---- Multi-Layer Perceptron script ----
# This file is intentionally left as a stub after extracting both ablation and baseline versions.
# Please use script_mlp_modified_ablation.py or script_mlp_modified_baseline.py for your experiments.
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
>>>>>>> e0998a4 (Baseline commit before ablation studies (original MLP architecture, results, and scripts))


    # Save results (evaluate on test set again to ensure predictions are up to date)
    model.eval()
    with torch.no_grad():
        final_test_outputs = model(X_test_tensor)
        _, final_test_pred = torch.max(final_test_outputs, 1)
        result_obj.data = final_test_pred.numpy().tolist()
    result_obj.fold_count = 0  # Ensure filename is valid
    import os
    os.makedirs(result_obj.result_destination_folder_path, exist_ok=True)
    result_obj.save()

<<<<<<< HEAD
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
=======
    # Calculate and print Precision, Recall, F1 Score
    from sklearn.metrics import precision_score, recall_score, f1_score
    y_true = y_test_tensor.numpy()
    y_pred = final_test_pred.numpy()
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")

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
>>>>>>> e0998a4 (Baseline commit before ablation studies (original MLP architecture, results, and scripts))
