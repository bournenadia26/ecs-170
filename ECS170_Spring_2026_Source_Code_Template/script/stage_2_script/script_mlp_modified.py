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

#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    script_dir = Path(__file__).resolve().parent

    data_obj = Dataset_Loader('MNIST', '')
    data_obj.dataset_source_folder_path = str(script_dir) + '/'
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
    tune_mlp(loaded_data)
    print('----- END -----')
    # ------------------------------------------------------
    

    