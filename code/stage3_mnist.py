'''
Stage 3 - MNIST Experiment Runner
ECS 170 Spring 2026

Runs CNN on MNIST hand-written digit dataset.
Also runs ablation experiments varying architecture configurations.
'''

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from dataset_loader.Dataset_Loader_MNIST import Dataset_Loader_MNIST
from method.Method_CNN_MNIST import Method_CNN_MNIST
from evaluate.Evaluate_Metrics import Evaluate_Metrics_CNN
from result.Result_Saver import Result_Saver_CNN
from Setting import Setting

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'stage_3_data', 'stage_3_data', 'MNIST')
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..', 'result_plots')


def run_experiment(config_name, max_epoch=10, lr=1e-3, dropout=0.25, batch_size=64):
    print(f'\n{"#"*60}')
    print(f'  MNIST Experiment: {config_name}')
    print(f'  epochs={max_epoch} | lr={lr} | dropout={dropout} | batch={batch_size}')
    print(f'{"#"*60}')

    # Dataset
    loader = Dataset_Loader_MNIST(dName='MNIST')
    loader.data_path  = DATA_PATH
    loader.batch_size = batch_size

    # Model
    model = Method_CNN_MNIST(mName=f'CNN_MNIST_{config_name}', dropout=dropout)
    model.max_epoch    = max_epoch
    model.learning_rate = lr

    # Evaluate & Save
    evaluator = Evaluate_Metrics_CNN()
    saver     = Result_Saver_CNN()
    saver.result_destination_folder_path = RESULT_DIR
    saver.result_destination_file_name   = config_name

    setting = Setting(sName=f'MNIST_{config_name}')
    setting.dataset      = loader
    setting.method       = model
    setting.evaluate     = evaluator
    setting.result_saver = saver

    metrics = setting.load_run_save_evaluate()
    return metrics


if __name__ == '__main__':
    results = {}

    # ---- Default configuration ----
    results['default'] = run_experiment(
        config_name='MNIST_default',
        max_epoch=10, lr=1e-3, dropout=0.25, batch_size=256
    )

    # ---- Ablation 1: higher dropout (more regularization) ----
    results['high_dropout'] = run_experiment(
        config_name='MNIST_high_dropout',
        max_epoch=10, lr=1e-3, dropout=0.5, batch_size=256
    )

    # ---- Ablation 2: lower learning rate ----
    results['low_lr'] = run_experiment(
        config_name='MNIST_low_lr',
        max_epoch=10, lr=3e-4, dropout=0.25, batch_size=256
    )

    # ---- Summary ----
    print('\n' + '='*60)
    print('  MNIST FINAL SUMMARY')
    print('='*60)
    for cfg, m in results.items():
        print(f'  {cfg:20s} | Acc: {m["accuracy"]:.4f} | F1(macro): {m["f1_macro"]:.4f}')
    print('='*60)
