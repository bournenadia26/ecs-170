
import sys
from pathlib import Path
# Ensure imports work no matter where the script is launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_code.stage_1_code.Result_Loader import Result_Loader

from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

def load_true_labels():
    # Assumes last column is the label, as in toy_data_file.txt
    y = []
    toy_data_path = PROJECT_ROOT / 'data' / 'stage_1_data' / 'toy_data_file.txt'
    with open(toy_data_path, 'r') as f:
        for line in f:
            elements = [int(i) for i in line.strip().split()]
            y.append(elements[-1])
    return np.array(y)

if 1:
    result_obj = Result_Loader('saver', '')
    result_obj.result_destination_folder_path = str(PROJECT_ROOT / 'result' / 'stage_1_result' / 'DT_')
    result_obj.result_destination_file_name = 'prediction_result'

    true_labels = load_true_labels()
    for fold_count in [1, 2, 3, None]:
        result_obj.fold_count = fold_count
        result_obj.load()
        # Expecting result_obj.data to be a dict with 'pred_y' and 'true_y'
        if isinstance(result_obj.data, dict) and 'pred_y' in result_obj.data and 'true_y' in result_obj.data:
            pred = np.array(result_obj.data['pred_y'])
            y_true = np.array(result_obj.data['true_y'])
            print('Fold:', fold_count, ', Result:', pred)
            if len(pred) == len(y_true):
                precision = precision_score(y_true, pred, average='weighted', zero_division=0)
                recall = recall_score(y_true, pred, average='weighted', zero_division=0)
                f1 = f1_score(y_true, pred, average='weighted', zero_division=0)
                print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
            else:
                print('Warning: Prediction and true label lengths do not match for this fold.')
        else:
            print('Fold:', fold_count, ', Result:', result_obj.data)
            print('Warning: Result data is not in expected format.')