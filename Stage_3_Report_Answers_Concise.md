# ECS 170 Spring 2026 - Stage 3 Report (Concise Version)

## Team Information
- Team Name: [Enter Team Name]
- Student 1: [Name] | [ID] | [Email]
- Student 2: [Name] | [ID] | [Email]
- Student 3: [Name] | [ID] | [Email]
- Student 4: [Name] | [ID] | [Email]
- Student 5: [Name] | [ID] | [Email]

## 1) Task Description
This stage focuses on image classification using CNN-based models across MNIST, ORL, and CIFAR-10. The objective is to train dataset-appropriate architectures, evaluate performance with standard metrics (Accuracy/Precision/Recall/F1), and study sensitivity to architectural and hyperparameter changes via ablations.

## 2) Model Description
- MNIST model: 2-layer CNN (Conv-BN-ReLU-Pool x2) + FC classifier.
- ORL model: deeper grayscale CNN with 3 conv stages + FC(512) + FC(40).
- CIFAR model: ResNet-18 adapted for 32x32 images (no initial maxpool), residual BasicBlocks, AdaptiveAvgPool, dropout head.

Final CIFAR training used MixUp, SGD+momentum+Nesterov, OneCycleLR, label smoothing, and standard CIFAR augmentations.

## 3) Experiment Settings

### 3.1 Dataset Description
- MNIST: 60,000 train / 10,000 test, 28x28 grayscale, 10 classes.
- ORL: 360 train / 40 test, 112x92 grayscale, 40 classes.
- CIFAR-10: 50,000 train / 10,000 test, 32x32 RGB, 10 classes.

### 3.2 Detailed Setup
- Framework: PyTorch.
- MNIST: Adam + StepLR, 10 epochs, ablations on dropout and lr.
- ORL: Adam + StepLR, 30-50 epochs, ablations on epochs/lr/dropout.
- CIFAR: ResNet-18, SGD(momentum=0.9, nesterov=True, weight_decay=5e-4), OneCycleLR, label smoothing=0.1, MixUp(alpha=0.2), 50 epochs.

### 3.3 Evaluation Metrics
Metrics used: Accuracy, Precision (macro/weighted), Recall (macro/weighted), F1 (macro/weighted/micro).

### 3.4 Source Code
- Local code path: `/Users/kavosh/Desktop/stage3/code`
- Public repository/link: [Add link]

### 3.5 Training Convergence Plot
All plot images are organized in `assets/plots/`:
- CIFAR_default_learning_curves.png
- CIFAR_high_dropout_learning_curves.png
- CIFAR_low_lr_learning_curves.png
- MNIST_MNIST_default_learning_curves.png
- MNIST_MNIST_high_dropout_learning_curves.png
- MNIST_MNIST_low_lr_learning_curves.png
- ORL_ORL_default_learning_curves.png
- ORL_ORL_more_epochs_learning_curves.png
- ORL_ORL_high_lr_learning_curves.png
- ORL_ORL_high_dropout_learning_curves.png

### 3.6 Model Performance

#### MNIST
| Config | Accuracy | Precision (Macro) | Recall (Macro) | F1 (Macro) |
|---|---:|---:|---:|---:|
| default | 0.9928 | 0.9927 | 0.9927 | 0.9927 |
| high_dropout | 0.9918 | 0.9918 | 0.9917 | 0.9917 |
| low_lr | 0.9904 | 0.9903 | 0.9903 | 0.9903 |

#### ORL
| Config | Accuracy | Precision (Macro) | Recall (Macro) | F1 (Macro) |
|---|---:|---:|---:|---:|
| default | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| more_epochs | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| high_lr | 0.9000 | 0.8500 | 0.9000 | 0.8667 |
| high_dropout | 0.9750 | 0.9625 | 0.9750 | 0.9667 |

#### CIFAR-10 (Uploaded Final Run)
| Config | Final Accuracy | Best Accuracy (Epoch) | Precision (Macro) | Recall (Macro) | F1 (Macro) | Train Time |
|---|---:|---:|---:|---:|---:|---:|
| default | 0.9462 | 0.9466 (48) | 0.9463 | 0.9462 | 0.9462 | 41.2 min |
| high_dropout | 0.9456 | 0.9470 (49) | 0.9459 | 0.9456 | 0.9457 | 40.9 min |
| low_lr | 0.8288 | 0.8300 (48) | 0.8275 | 0.8288 | 0.8278 | 41.0 min |

### 3.7 Ablation Studies
- MNIST: stronger dropout or lower lr slightly decreases performance due to mild underfitting in fixed epoch budget.
- ORL: higher lr hurts stability (F1 drop), while extra epochs do not improve beyond saturation.
- CIFAR: low lr underfits strongly (~82.9%), while default and high-dropout both converge around ~94.6-94.7% best accuracy with stable late epochs.

## Conclusion
Dataset-specific model capacity and optimization are critical. The final pipeline achieved strong and stable performance across datasets, with best CIFAR performance around 94.7% and robust macro-F1.
