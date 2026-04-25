# ECS170 Spring 2026 Machine Learning Project

## Project Objective
This project is designed to help you understand and experiment with basic machine learning classification algorithms using a small, interpretable dataset. You will:
- Learn how to implement and evaluate Decision Tree, Multi-Layer Perceptron (MLP), and Support Vector Machine (SVM) classifiers.
- Practice using cross-validation and accuracy evaluation.
- Gain experience with Python machine learning libraries (scikit-learn, PyTorch).

## Dataset
- **Location:** `ECS170_Spring_2026_Source_Code_Template/data/stage_1_data/toy_data_file.txt`
- **Description:** 16 rows, each with 4 features and 1 label (binary classification).
- **Format:** Each line: `f1 f2 f3 f4 label`

## Project Structure
- **data/**: Contains the dataset.
- **local_code/**: Core code for data loading, model methods, evaluation, and settings.
- **script/stage_1_script/**: Scripts to run each model (Decision Tree, MLP, SVM) and to load results.
- **result/**: Where prediction results are saved after running scripts.

## How It Works
1. **Scripts** (in `script/stage_1_script/`) load the dataset, initialize a model, set up cross-validation, train, evaluate, and save results.
2. **Methods** (in `local_code/stage_1_code/Method_*.py`) define how each model is trained and tested.
3. **Evaluation** uses accuracy as the metric.

## How to Complete the Project
1. **Install dependencies:**
   - Make sure you have Python 3.10+ and install required packages:
     ```
     pip install numpy scikit-learn torch
     ```
2. **Run the scripts:**
   - From the project root, run:
     ```
     cd ECS170_Spring_2026_Source_Code_Template
     python -m script.stage_1_script.script_decision_tree
     python -m script.stage_1_script.script_mlp
     python -m script.stage_1_script.script_svm
     ```
   - This will train and evaluate each model using K-Fold cross-validation and save results.
3. **Check results:**
   - Results are saved in `result/stage_1_result/`.
   - To print results, run:
     ```
     python -m script.stage_1_script.script_load_result
     ```
4. **(Optional) Modify/extend:**
   - You can change model parameters, try different settings, or use your own data.

## Learning Goals
- Understand the workflow of a machine learning experiment.
- Learn how to use and compare different classifiers.
- Practice with cross-validation and result saving/loading.

---
**If you have any issues running the scripts, check the import paths or let your instructor/TA know.**
