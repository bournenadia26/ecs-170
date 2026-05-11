# ECS 170 Spring 2026 - Stage 3 Report

## Team Information
- Team Name: [Enter Team Name]
- Student 1: [Name] | [ID] | [Email]
- Student 2: [Name] | [ID] | [Email]
- Student 3: [Name] | [ID] | [Email]
- Student 4: [Name] | [ID] | [Email]
- Student 5: [Name] | [ID] | [Email]

## Section 1: Task Description
In Stage 3, we study supervised image classification using convolutional neural networks (CNNs). We trained and evaluated models on three datasets with different complexity levels: MNIST (digits), ORL (faces), and CIFAR-10 (natural objects). The goal is to build effective CNN pipelines, evaluate them with standard classification metrics, and analyze the effect of architectural and hyperparameter changes through ablation studies.

## Section 2: Model Description
We used dataset-specific CNN designs.

1. MNIST model (2-layer CNN)
- Conv(1->32) + BatchNorm + ReLU + MaxPool
- Conv(32->64) + BatchNorm + ReLU + MaxPool
- Dropout2d
- FC(3136->128) + ReLU + Dropout
- FC(128->10)

2. ORL model (deeper grayscale CNN)
- Three convolutional blocks with BatchNorm/ReLU/Pooling
- FC(19712->512)
- FC(512->40)

3. CIFAR-10 model (ResNet-18, CIFAR-adapted)
- Stem: Conv(3->64, 3x3, stride 1), BatchNorm, ReLU (no initial maxpool)
- Residual stages with BasicBlocks: 64, 128, 256, 512 channels
- Adaptive average pooling + dropout + FC(512->10)

For final CIFAR training, we used MixUp regularization, SGD with momentum+Nesterov, OneCycleLR scheduling, and label smoothing.

## Section 3: Experiment Settings

### 3.1 Dataset Description
- MNIST: 60,000 training, 10,000 testing; 28x28 grayscale; 10 classes.
- ORL: 360 training, 40 testing; 112x92 grayscale; 40 classes.
- CIFAR-10: 50,000 training, 10,000 testing; 32x32 RGB; 10 classes.

We used the provided train/test split for each dataset.

### 3.2 Detailed Experimental Setups
Common:
- Framework: PyTorch
- Evaluation: Accuracy, Precision, Recall, F1 (macro/weighted/micro)

MNIST:
- Optimizer: Adam
- Scheduler: StepLR(step=5, gamma=0.5)
- Epochs: 10
- Ablations: dropout and learning rate

ORL:
- Optimizer: Adam
- Scheduler: StepLR(step=10, gamma=0.5)
- Epochs: 30 (default), 50 (ablation)
- Ablations: epochs, learning rate, dropout

CIFAR-10 (final uploaded run):
- Model: ResNet-18 (CIFAR adapted)
- Optimizer: SGD(momentum=0.9, nesterov=True, weight_decay=5e-4)
- Scheduler: OneCycleLR (cosine annealing)
- Loss: CrossEntropy with label smoothing=0.1
- Regularization: MixUp(alpha=0.2), dropout (ablation)
- Epochs: 50
- Batch size: 256
- Augmentation: RandomCrop, RandomHorizontalFlip, ColorJitter, Normalize, RandomErasing

### 3.3 Evaluation Metrics
- Accuracy: ratio of correct predictions over all predictions.
- Precision: ratio of true positives over predicted positives.
- Recall: ratio of true positives over actual positives.
- F1 score: harmonic mean of precision and recall.

Macro averages treat all classes equally. Weighted averages account for class frequency.

### 3.4 Source Code
- Local code path: `/Users/kavosh/Desktop/stage3/code`
- Public repository or shared link: [Add link]

### 3.5 Training Convergence Plot
All convergence plots are prepared in:
- `assets/plots/`

Included files:
- `MNIST_MNIST_default_learning_curves.png`
- `MNIST_MNIST_high_dropout_learning_curves.png`
- `MNIST_MNIST_low_lr_learning_curves.png`
- `ORL_ORL_default_learning_curves.png`
- `ORL_ORL_more_epochs_learning_curves.png`
- `ORL_ORL_high_lr_learning_curves.png`
- `ORL_ORL_high_dropout_learning_curves.png`
- `CIFAR_default_learning_curves.png`
- `CIFAR_high_dropout_learning_curves.png`
- `CIFAR_low_lr_learning_curves.png`

### 3.6 Model Performance

MNIST:

| Config | Accuracy | Precision (Macro) | Recall (Macro) | F1 (Macro) |
|---|---:|---:|---:|---:|
| default | 0.9928 | 0.9927 | 0.9927 | 0.9927 |
| high_dropout | 0.9918 | 0.9918 | 0.9917 | 0.9917 |
| low_lr | 0.9904 | 0.9903 | 0.9903 | 0.9903 |

ORL:

| Config | Accuracy | Precision (Macro) | Recall (Macro) | F1 (Macro) |
|---|---:|---:|---:|---:|
| default | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| more_epochs | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| high_lr | 0.9000 | 0.8500 | 0.9000 | 0.8667 |
| high_dropout | 0.9750 | 0.9625 | 0.9750 | 0.9667 |

CIFAR-10 (uploaded final run):

| Config | Final Accuracy | Best Accuracy (Epoch) | Precision (Macro) | Recall (Macro) | F1 (Macro) | Train Time |
|---|---:|---:|---:|---:|---:|---:|
| default | 0.9462 | 0.9466 (48) | 0.9463 | 0.9462 | 0.9462 | 41.2 min |
| high_dropout | 0.9456 | 0.9470 (49) | 0.9459 | 0.9456 | 0.9457 | 40.9 min |
| low_lr | 0.8288 | 0.8300 (48) | 0.8275 | 0.8288 | 0.8278 | 41.0 min |

### 3.7 Ablation Studies
MNIST:
- Increasing dropout and lowering LR both slightly reduced performance under the same 10-epoch budget.

ORL:
- More epochs did not improve beyond saturation (already 100% accuracy).
- High learning rate reduced macro-F1 significantly.
- High dropout mildly reduced performance versus default.

CIFAR-10:
- Default and high-dropout settings both reached strong performance (~94.6-94.7% best).
- Low learning rate underfit severely (~83%), indicating insufficient optimization aggressiveness for this schedule.
- Best and final accuracy are close for strong configs, showing stable late-epoch convergence.

## Conclusion
The experiments show that model complexity and optimization strategy must match dataset difficulty. A compact CNN is sufficient for MNIST, a deeper CNN works well for ORL, and a residual architecture with strong regularization/scheduling is necessary for CIFAR-10. The best CIFAR result reached 94.70% test accuracy (best epoch), with 94.62% final accuracy.
