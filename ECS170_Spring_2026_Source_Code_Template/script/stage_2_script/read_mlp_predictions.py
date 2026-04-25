import pickle

<<<<<<< HEAD
# Path to the result file
target_file = "../../result/stage_2_result/MLP_prediction_result_0"
=======

# Path to the result file (absolute path for reliability)
import os
target_file = os.path.join(os.path.dirname(__file__), '../../result/stage_2_result/MLP_prediction_result_0')
target_file = os.path.abspath(target_file)
>>>>>>> 74723fc (Baseline commit before ablation studies (original MLP architecture, results, and scripts))

with open(target_file, "rb") as f:
    predictions = pickle.load(f)

<<<<<<< HEAD
=======
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# Load true labels from test.csv
test_file = os.path.join(os.path.dirname(__file__), '../../data/stage_2_data/test.csv')
test_file = os.path.abspath(test_file)
true_labels = []
with open(test_file, "r") as f:
    for line in f:
        true_labels.append(int(line.strip().split(',')[0]))

# Compute metrics
y_true = np.array(true_labels)
y_pred = np.array(predictions)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1 Score: {f1:.4f}")
>>>>>>> 74723fc (Baseline commit before ablation studies (original MLP architecture, results, and scripts))
print("Predictions:")
print(predictions)
print(f"Total predictions: {len(predictions)}")
