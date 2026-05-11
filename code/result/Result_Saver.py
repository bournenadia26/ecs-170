'''
Result Saver for ECS 170 Stage 3
Saves learning curves (loss & accuracy) and metrics to disk.
'''

import os
import json
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from base.base_result_saver import Result_Saver


class Result_Saver_CNN(Result_Saver):
    def __init__(self, sName='Result_Saver_CNN', sDescription='Save CNN results and plots'):
        super().__init__(sName=sName, sDescription=sDescription)
        self.result_destination_folder_path = '../result_plots'
        self.result_destination_file_name = 'results'

    def save(self, raw_result, dataset_name=''):
        '''
        raw_result dict should contain:
          'loss_history', 'acc_history', 'metrics', 'pred_y', 'true_y'
        '''
        os.makedirs(self.result_destination_folder_path, exist_ok=True)
        prefix = os.path.join(self.result_destination_folder_path,
                              f'{dataset_name}_{self.result_destination_file_name}')

        # ---- Plot learning curves ----
        epochs = list(range(1, len(raw_result['loss_history']) + 1))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(epochs, raw_result['loss_history'], 'b-o', markersize=4, label='Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Cross-Entropy Loss')
        ax1.set_title(f'{dataset_name} — Training Loss Curve')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(epochs, raw_result['acc_history'], 'g-o', markersize=4, label='Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title(f'{dataset_name} — Training Accuracy Curve')
        ax2.set_ylim([0, 1.05])
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plot_path = f'{prefix}_learning_curves.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f'  Saved learning curve plot: {plot_path}')

        # ---- Save metrics JSON ----
        metrics_path = f'{prefix}_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(raw_result['metrics'], f, indent=2)
        print(f'  Saved metrics JSON: {metrics_path}')
