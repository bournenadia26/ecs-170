import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN import Method_CNN

np.random.seed(2)
torch.manual_seed(2)

# load MNIST once and reuse across all configs
loader = Dataset_Loader(
    dataset_name='MNIST',
    dataset_path=str(PROJECT_ROOT / 'data' / 'stage_3_data' / 'MNIST')
)
data = loader.load()

# subsample training set so ablation finishes in reasonable time on CPU
N_TRAIN = 10000
data['train']['X'] = data['train']['X'][:N_TRAIN]
data['train']['y'] = data['train']['y'][:N_TRAIN]

configs = [
    {'name': 'Baseline',       'num_filters': 32, 'fc_hidden_size': 256, 'dropout_rate': 0.0, 'num_conv_layers': 2},
    {'name': 'Fewer Filters',  'num_filters': 16, 'fc_hidden_size': 256, 'dropout_rate': 0.0, 'num_conv_layers': 2},
    {'name': 'More Filters',   'num_filters': 64, 'fc_hidden_size': 256, 'dropout_rate': 0.0, 'num_conv_layers': 2},
    {'name': 'Smaller FC',     'num_filters': 32, 'fc_hidden_size': 128, 'dropout_rate': 0.0, 'num_conv_layers': 2},
    {'name': 'With Dropout',   'num_filters': 32, 'fc_hidden_size': 256, 'dropout_rate': 0.3, 'num_conv_layers': 2},
    {'name': '3 Conv Layers',  'num_filters': 32, 'fc_hidden_size': 256, 'dropout_rate': 0.0, 'num_conv_layers': 3},
]

results = []

for cfg in configs:
    print(f"\n{'='*60}")
    print(f"Running config: {cfg['name']}")
    print(f"  filters={cfg['num_filters']}, fc_hidden={cfg['fc_hidden_size']}, "
          f"dropout={cfg['dropout_rate']}, conv_layers={cfg['num_conv_layers']}")
    print('='*60)

    model = Method_CNN('cnn', '', in_channels=1, img_size=28, num_classes=10)
    model._build_model(
        num_filters=cfg['num_filters'],
        fc_hidden_size=cfg['fc_hidden_size'],
        dropout_rate=cfg['dropout_rate'],
        num_conv_layers=cfg['num_conv_layers']
    )
    model.max_epoch = 30
    model.learning_rate = 1e-3
    model.data = data

    run_result = model.run()

    pred_y = run_result['pred_y'].numpy()
    true_y = np.array(run_result['true_y'])

    accuracy  = np.mean(pred_y == true_y)
    precision = precision_score(true_y, pred_y, average='weighted', zero_division=0)
    recall    = recall_score(true_y, pred_y, average='weighted', zero_division=0)
    f1        = f1_score(true_y, pred_y, average='weighted', zero_division=0)
    final_test_loss = run_result['test_loss_history'][-1]

    results.append({
        'config': cfg,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'final_test_loss': final_test_loss,
        'loss_history': run_result['loss_history'],
        'acc_history': run_result['acc_history'],
        'test_loss_history': run_result['test_loss_history'],
        'test_acc_history': run_result['test_acc_history'],
    })

    print(f"  Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | "
          f"Recall: {recall:.4f} | F1: {f1:.4f} | Test Loss: {final_test_loss:.4f}")

result_dir = PROJECT_ROOT / 'result' / 'stage_3_result'
os.makedirs(result_dir, exist_ok=True)

# save results table
with open(result_dir / 'cnn_ablation_results.txt', 'w') as f:
    f.write('----- CNN STRUCTURE ABLATION RESULTS (MNIST) -----\n\n')
    header = f"{'Config':<18} {'Filters':>8} {'FC Hidden':>10} {'Dropout':>8} {'Conv Layers':>12} {'Accuracy':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Test Loss':>10}\n"
    f.write(header)
    f.write('-' * len(header) + '\n')
    for r in results:
        cfg = r['config']
        f.write(
            f"{cfg['name']:<18} {cfg['num_filters']:>8} {cfg['fc_hidden_size']:>10} "
            f"{cfg['dropout_rate']:>8.1f} {cfg['num_conv_layers']:>12} "
            f"{r['accuracy']:>10.4f} {r['precision']:>10.4f} {r['recall']:>8.4f} "
            f"{r['f1']:>8.4f} {r['final_test_loss']:>10.4f}\n"
        )

print('\n----- CNN ABLATION SUMMARY -----')
for r in results:
    print(f"  {r['config']['name']:<18} Accuracy={r['accuracy']:.4f}  F1={r['f1']:.4f}  TestLoss={r['final_test_loss']:.4f}")

# plot learning curves — 2 rows x 3 cols
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, r in enumerate(results):
    ax = axes[i]
    epochs = range(1, len(r['loss_history']) + 1)
    ax.plot(epochs, r['loss_history'], label='Train Loss')
    ax.plot(epochs, r['test_loss_history'], label='Test Loss')
    ax2 = ax.twinx()
    ax2.plot(epochs, r['acc_history'], label='Train Acc', linestyle='--', color='green', alpha=0.6)
    ax2.plot(epochs, r['test_acc_history'], label='Test Acc', linestyle='--', color='orange', alpha=0.6)
    ax2.set_ylabel('Accuracy')
    ax.set_title(r['config']['name'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(loc='upper left', fontsize=7)
    ax2.legend(loc='upper right', fontsize=7)

plt.suptitle('CNN Structure Ablation — MNIST Learning Curves', fontsize=13)
plt.tight_layout()
plt.savefig(result_dir / 'cnn_ablation_curves.png')
plt.show()
print(f'\nResults saved to {result_dir}')
