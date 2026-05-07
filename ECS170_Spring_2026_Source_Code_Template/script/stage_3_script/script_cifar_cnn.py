import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN import Method_CNN

# load dataset
loader = Dataset_Loader(
    dataset_name='CIFAR',
    dataset_path='../../data/stage_3_data/CIFAR'
)

data = loader.load()

# check dataset
print(data['train']['X'].shape)
print(data['test']['X'].shape)
print(data['train']['y'][:10])
print(data['test']['y'][:10])

# run tuning
result = Method_CNN.tune_cnn(
    data,
    in_channels=3,
    img_size=32,
    num_classes=10
)

# best config
best_config = result['best_config']
best_accuracy = result['best_accuracy']

# evaluation metrics
pred_y = result['pred_y'].numpy()
true_y = np.array(result['true_y'])

accuracy = np.mean(pred_y == true_y)

precision = precision_score(
    true_y,
    pred_y,
    average='weighted',
    zero_division=0
)

recall = recall_score(
    true_y,
    pred_y,
    average='weighted',
    zero_division=0
)

f1 = f1_score(
    true_y,
    pred_y,
    average='weighted',
    zero_division=0
)

final_test_loss = result['test_loss_history'][-1]

# print results
print('----- FINAL CIFAR RESULTS -----')
print('Best Config:', best_config)
print('Best Accuracy:', best_accuracy)
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1:', f1)
print('Test Loss:', final_test_loss)

# save results and figures
result_dir = '../../result/stage_3_result'
os.makedirs(result_dir, exist_ok=True)

with open(os.path.join(result_dir, 'cifar_results.txt'), 'w') as f:
    f.write('----- FINAL CIFAR RESULTS -----\n')
    f.write(f'Best Config: {best_config}\n')
    f.write(f'Best Accuracy: {best_accuracy}\n')
    f.write(f'Accuracy: {accuracy}\n')
    f.write(f'Precision: {precision}\n')
    f.write(f'Recall: {recall}\n')
    f.write(f'F1: {f1}\n')
    f.write(f'Test Loss: {final_test_loss}\n')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(result['loss_history'], label='Train Loss')
axes[0].plot(result['test_loss_history'], label='Test Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('CIFAR Loss Curve')
axes[0].legend()

axes[1].plot(result['acc_history'], label='Train Accuracy')
axes[1].plot(result['test_acc_history'], label='Test Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('CIFAR Accuracy Curve')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'cifar_training_curves.png'))
plt.show()