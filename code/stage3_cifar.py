'''
Stage 3 - CIFAR-10 Experiment Runner
ECS 170 Spring 2026

Runs CNN on CIFAR-10 colored object dataset.
Also runs ablation experiments varying architecture configurations.
'''

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset_loader.Dataset_Loader_CIFAR import Dataset_Loader_CIFAR
from method.Method_CNN_CIFAR import Method_CNN_CIFAR
from evaluate.Evaluate_Metrics import Evaluate_Metrics_CNN
from result.Result_Saver import Result_Saver_CNN
from Setting import Setting

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'stage_3_data', 'stage_3_data', 'CIFAR')
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..', 'result_plots')


def run_experiment(config_name, max_epoch=60, lr=1e-3, dropout=0.0, batch_size=512):
    print(f'\n{"#"*60}')
    print(f'  CIFAR-10 Experiment: {config_name}')
    print(f'  epochs={max_epoch} | lr={lr} | dropout={dropout} | batch={batch_size}')
    print(f'{"#"*60}')

    loader = Dataset_Loader_CIFAR(dName='CIFAR-10')
    loader.data_path  = DATA_PATH
    loader.batch_size = batch_size

    model = Method_CNN_CIFAR(mName=f'CNN_CIFAR_{config_name}', dropout=dropout)
    model.max_epoch     = max_epoch
    model.learning_rate = lr

    evaluator = Evaluate_Metrics_CNN()
    saver     = Result_Saver_CNN()
    saver.result_destination_folder_path = RESULT_DIR
    saver.result_destination_file_name   = config_name

    setting = Setting(sName=f'CIFAR_{config_name}')
    setting.dataset      = loader
    setting.method       = model
    setting.evaluate     = evaluator
    setting.result_saver = saver

    metrics = setting.load_run_save_evaluate()
    return metrics


if __name__ == '__main__':
    results = {}

    # ---- Default: ResNet-18 + full augmentation (target 90%+) ----
    results['default'] = run_experiment(
        config_name='CIFAR_default',
        max_epoch=30, lr=1e-3, dropout=0.0, batch_size=512
    )

    # ---- Ablation 1: with dropout ----
    results['high_dropout'] = run_experiment(
        config_name='CIFAR_high_dropout',
        max_epoch=30, lr=1e-3, dropout=0.3, batch_size=512
    )

    # ---- Ablation 2: lower learning rate ----
    results['low_lr'] = run_experiment(
        config_name='CIFAR_low_lr',
        max_epoch=30, lr=3e-4, dropout=0.0, batch_size=512
    )

    # ---- Summary ----
    print('\n' + '='*60)
    print('  CIFAR-10 FINAL SUMMARY')
    print('='*60)
    for cfg, m in results.items():
        print(f'  {cfg:20s} | Acc: {m["accuracy"]:.4f} | F1(macro): {m["f1_macro"]:.4f}')
    print('='*60)
