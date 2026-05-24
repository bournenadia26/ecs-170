from pathlib import Path
import sys

# Ensure imports work no matter where the script is launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP_ablation import Method_MLP
from local_code.stage_1_code.Result_Saver import Result_Saver
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
    # Use robust absolute path for result saving
    result_dir = PROJECT_ROOT / 'result' / 'stage_2_result'
    result_obj.result_destination_folder_path = str(result_dir / 'MLP_')
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test_Split('train test split', '')
    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # load data
    loaded_data = data_obj.load()

    # Example: train and evaluate MLP
    print('----- BEGIN TRAINING -----')
    X_train = np.array(loaded_data['train']['X'], dtype=np.float32)
    y_train = np.array(loaded_data['train']['y'], dtype=np.int64)
    X_test = np.array(loaded_data['test']['X'], dtype=np.float32)
    y_test = np.array(loaded_data['test']['y'], dtype=np.int64)

    # plotting
    # If you have tuning results, you can plot them here
    # ...existing code...
