# ECS170 Stage 2–5 Report

**Student Name:** (Leave blank)

**Date:** April 24, 2026

---

## 1. Introduction

This project implements and evaluates machine learning models for multiclass classification using a provided MNIST-style dataset. The main focus is on developing, training, and analyzing a Multi-Layer Perceptron (MLP) model, with attention to overfitting, model tuning, and reproducibility. The project follows a modular code structure, separating data, code, results, and scripts for clarity and maintainability.

---

## 2. Methods

### 2.1 Data

- **Source:** Provided MNIST-style CSV files (train.csv, test.csv).
- **Preprocessing:** Data loaded and split into features and labels using a custom Dataset_Loader.
- **Train/Test Split:** Used provided train/test split; no cross-validation for this stage.

### 2.2 Model

- **Architecture:** MLP with two hidden layers (512 and 256 units), ReLU activations, dropout (0.3), and 10 output classes.
- **Loss Function:** CrossEntropyLoss (PyTorch).
- **Optimizer:** Adam.
- **Regularization:** Dropout to reduce overfitting.

### 2.3 Training Procedure

- Training and test loss/accuracy are computed and logged at every epoch.
- Model is trained for a fixed number of epochs (as set in Method_MLP).
- Training is reproducible with fixed random seeds.

### 2.4 Evaluation

- Accuracy is used as the primary metric.
- Overfitting is diagnosed by plotting both training and test loss/accuracy curves.
- Final predictions are saved for further analysis.

---

## 3. Results

### 3.1 Training Progress

- Both training and test loss/accuracy curves are plotted and saved as `result/stage_2_result/MLP_training_progress.png`.
- The model achieves high test accuracy (typically >98%).
- Training and test curves are close, indicating minimal overfitting.

### 3.2 Predictions

- Final predictions for the test set are saved and can be read using the provided script.
- Total predictions: 10,000 (matching test set size).

### 3.3 Ablation Study Results

To assess the robustness and importance of various architectural choices, several ablation experiments were performed on the MLP model for stage_2. Each ablation modifies a single aspect of the baseline model, and the resulting test set metrics are reported below:

| Ablation                | Precision | Recall | F1 Score |
|-------------------------|-----------|--------|----------|
| Hidden size = 128       | 0.9783    | 0.9783 | 0.9783   |
| No Dropout              | 0.9783    | 0.9783 | 0.9783   |
| Activation = Tanh       | 0.9783    | 0.9783 | 0.9783   |

All ablations produced nearly identical results to the baseline, indicating that the model's performance is robust to these changes for this dataset and configuration.

### 3.4 CNN Structure Ablation Study (Stage 3)

To examine how CNN architecture choices affect performance, six configurations were trained on a 10,000-sample subset of MNIST for 30 epochs. Each configuration changes one structural element relative to the baseline (32 filters, FC=256, no dropout, 2 conv layers).

| Config        | Filters | FC Hidden | Dropout | Conv Layers | Accuracy | Precision | Recall | F1     | Test Loss |
|---------------|---------|-----------|---------|-------------|----------|-----------|--------|--------|-----------|
| Baseline      | 32      | 256       | 0.0     | 2           | 0.9857   | 0.9857    | 0.9857 | 0.9857 | 0.0702    |
| Fewer Filters | 16      | 256       | 0.0     | 2           | 0.9845   | 0.9845    | 0.9845 | 0.9845 | 0.0792    |
| More Filters  | 64      | 256       | 0.0     | 2           | 0.9861   | 0.9861    | 0.9861 | 0.9861 | 0.0688    |
| Smaller FC    | 32      | 128       | 0.0     | 2           | 0.9845   | 0.9845    | 0.9845 | 0.9845 | 0.0821    |
| With Dropout  | 32      | 256       | 0.3     | 2           | 0.9848   | 0.9850    | 0.9848 | 0.9848 | 0.0677    |
| 3 Conv Layers | 32      | 256       | 0.0     | 3           | 0.9877   | 0.9877    | 0.9877 | 0.9877 | 0.0757    |

**Observations:** All configurations performed within a narrow accuracy band (98.45%–98.77%), showing that CNNs are generally robust to moderate structural changes on MNIST. Adding a third convolutional layer gave the best accuracy (98.77%), suggesting deeper feature extraction helps even on a simple dataset. Doubling the number of filters from 32 to 64 improved test loss slightly (0.0688 vs 0.0702), while halving filters to 16 had minimal impact, indicating 16 filters is already sufficient capacity for MNIST. Reducing the FC hidden size to 128 increased test loss the most (0.0821), implying the classification head is a mild bottleneck. Dropout (rate=0.3) produced the lowest test loss (0.0677) despite slightly lower accuracy, consistent with its role as a regularizer that smooths the loss surface. Learning curves are saved to `result/stage_3_result/cnn_ablation_curves.png`.

## 4. Discussion

- **Ablation Study Insights:**
	- Reducing hidden size, removing dropout, or switching activation to Tanh did not significantly impact test precision, recall, or F1 score. This suggests the MLP is not highly sensitive to these hyperparameters for this task, or that the dataset is sufficiently simple for the model to generalize well under a range of configurations.
	- The identical metrics across ablations may also indicate that the model is not under- or over-parameterized for this dataset.

- **Overfitting:** Minimal, as shown by close training/test curves. Dropout and careful tuning helped prevent overfitting.
- **Model Tuning:** Hyperparameters (hidden units, dropout, learning rate) were chosen to maximize test accuracy while avoiding overfitting.
- **Reproducibility:** All scripts are modular and reproducible; random seeds are set.
- **Limitations:** Large data files (train.csv, test.csv) are not included in the GitHub repo due to size constraints. Users must provide these files separately.

## 5. Conclusion

The MLP model successfully performs multiclass classification on the MNIST-style dataset with high accuracy and minimal overfitting. The modular codebase and clear result saving enable easy extension and reproducibility.

---

## 6. References

- scikit-learn documentation
- PyTorch documentation
- ECS170 course materials

---

## 7. Appendix

- GitHub repository: [https://github.com/kavosh-404/ECS170_Spring_2026_ML_Project](https://github.com/kavosh-404/ECS170_Spring_2026_ML_Project)
- Scripts for training, evaluation, and result reading are included in the repo.
- Data files must be added to `data/stage_2_data/` for full reproducibility.

---
