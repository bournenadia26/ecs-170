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

---

## 4. Discussion

- **Overfitting:** Minimal, as shown by close training/test curves. Dropout and careful tuning helped prevent overfitting.
- **Model Tuning:** Hyperparameters (hidden units, dropout, learning rate) were chosen to maximize test accuracy while avoiding overfitting.
- **Reproducibility:** All scripts are modular and reproducible; random seeds are set.
- **Limitations:** Large data files (train.csv, test.csv) are not included in the GitHub repo due to size constraints. Users must provide these files separately.

---

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
