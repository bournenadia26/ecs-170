'''
Evaluate Metrics for ECS 170 Stage 3
Computes Accuracy, Precision, Recall, F1 (macro & weighted) for multiclass classification.
'''

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report)
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from base.base_evaluate import Evaluate_Metrics


class Evaluate_Metrics_CNN(Evaluate_Metrics):
    def __init__(self, eName='Evaluate_Metrics_CNN', eDescription='Multiclass CNN metrics'):
        super().__init__(eName=eName, eDescription=eDescription)

    def evaluate(self, pred_y, true_y):
        acc          = accuracy_score(true_y, pred_y)
        precision_m  = precision_score(true_y, pred_y, average='macro',    zero_division=0)
        precision_w  = precision_score(true_y, pred_y, average='weighted', zero_division=0)
        recall_m     = recall_score(true_y, pred_y, average='macro',    zero_division=0)
        recall_w     = recall_score(true_y, pred_y, average='weighted', zero_division=0)
        f1_m         = f1_score(true_y, pred_y, average='macro',    zero_division=0)
        f1_w         = f1_score(true_y, pred_y, average='weighted', zero_division=0)
        f1_micro     = f1_score(true_y, pred_y, average='micro',    zero_division=0)

        result = {
            'accuracy':           round(acc, 4),
            'precision_macro':    round(precision_m, 4),
            'precision_weighted': round(precision_w, 4),
            'recall_macro':       round(recall_m, 4),
            'recall_weighted':    round(recall_w, 4),
            'f1_macro':           round(f1_m, 4),
            'f1_weighted':        round(f1_w, 4),
            'f1_micro':           round(f1_micro, 4),
        }
        return result

    def print_report(self, pred_y, true_y, dataset_name=''):
        result = self.evaluate(pred_y, true_y)
        print(f'\n{"="*55}')
        print(f'  Evaluation Results: {dataset_name}')
        print(f'{"="*55}')
        print(f'  Accuracy:            {result["accuracy"]:.4f}')
        print(f'  Precision (macro):   {result["precision_macro"]:.4f}')
        print(f'  Precision (weighted):{result["precision_weighted"]:.4f}')
        print(f'  Recall (macro):      {result["recall_macro"]:.4f}')
        print(f'  Recall (weighted):   {result["recall_weighted"]:.4f}')
        print(f'  F1 (macro):          {result["f1_macro"]:.4f}')
        print(f'  F1 (weighted):       {result["f1_weighted"]:.4f}')
        print(f'  F1 (micro):          {result["f1_micro"]:.4f}')
        print(f'{"="*55}\n')
        return result
