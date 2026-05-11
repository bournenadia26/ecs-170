# Image Classification Using Convolutional Neural Networks (CNNs)

ECS 170 (Spring 2026) - Course Project Stage 3

This repository contains a complete Stage 3 pipeline for image classification on:
- MNIST
- ORL Faces
- CIFAR-10

It includes dataset loaders, CNN/ResNet models, training runners, evaluation metrics, result saving utilities, and report-ready plots/results.

## Repository Structure

- `code/`
  - `base/`: abstract base classes
  - `dataset_loader/`: loaders for MNIST, ORL, CIFAR
  - `method/`: model definitions (`Method_CNN_MNIST`, `Method_CNN_ORL`, `Method_CNN_CIFAR`)
  - `evaluate/`: evaluation metric implementation
  - `result/`: plotting and metric saving
  - `Setting.py`: train/eval orchestration
  - `stage3_mnist.py`, `stage3_orl.py`, `stage3_cifar.py`: experiment runners
- `result_plots/`: locally saved metric JSONs and learning curves
- `assets/plots/`: curated convergence plots for report insertion
- `CIFAR10_Colab.ipynb`: Colab notebook used for faster CIFAR training on T4 GPU
- `Stage_3_Report_Answers.md`: full report draft
- `Stage_3_Report_Answers_Concise.md`: concise report draft
- `Stage_3_Report_Submission_Ready.md`: final polished report draft

## Environment

Recommended:
- Python 3.10+
- PyTorch
- torchvision
- scikit-learn
- matplotlib

Install dependencies (example):

```bash
pip install torch torchvision scikit-learn matplotlib numpy
```

## Running Experiments

From project root:

### MNIST
```bash
cd code
python3 stage3_mnist.py
```

### ORL
```bash
cd code
python3 stage3_orl.py
```

### CIFAR-10 (local)
```bash
cd code
python3 stage3_cifar.py
```

Note: CIFAR training is significantly faster on Google Colab GPU using `CIFAR10_Colab.ipynb`.

## Final Performance (Key Results)

### MNIST
- default: Accuracy 0.9928, F1-macro 0.9927
- high_dropout: Accuracy 0.9918, F1-macro 0.9917
- low_lr: Accuracy 0.9904, F1-macro 0.9903

### ORL
- default: Accuracy 1.0000, F1-macro 1.0000
- more_epochs: Accuracy 1.0000, F1-macro 1.0000
- high_lr: Accuracy 0.9000, F1-macro 0.8667
- high_dropout: Accuracy 0.9750, F1-macro 0.9667

### CIFAR-10 (uploaded final run)
- default: Final Accuracy 0.9462 (best 0.9466 @ epoch 48)
- high_dropout: Final Accuracy 0.9456 (best 0.9470 @ epoch 49)
- low_lr: Final Accuracy 0.8288 (best 0.8300 @ epoch 48)

## Report Assets

Convergence plots for report use are available in:
- `assets/plots/`

Representative figures embedded in `Stage_3_Report_Answers.md` Section 3.5.

## Notes

- Large local files (dataset archives/venv) are excluded via `.gitignore`.
- If you run on Apple Silicon MPS, some optimization choices differ from CUDA behavior.

## Author

Project owner: `kavosh-404`
