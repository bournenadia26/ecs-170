import os
import sys
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ------------------------------------------------
# Project path setup
# ------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "../../")
)

STAGE4_CODE_DIR = os.path.join(
    PROJECT_ROOT,
    "local_code",
    "stage_4_code"
)

if STAGE4_CODE_DIR not in sys.path:
    sys.path.append(STAGE4_CODE_DIR)


from Dataset_Loader import load_classification_data, PAD_IDX
from rnn_classifier import RNNClassifier


# ------------------------------------------------
# Reproducibility
# ------------------------------------------------
def set_seed(seed=170):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------
# PyTorch Dataset
# ------------------------------------------------
class TextClassificationDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ------------------------------------------------
# Train one epoch
# ------------------------------------------------
def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    grad_clip=1.0
):
    model.train()

    total_loss = 0.0

    for texts, labels in dataloader:

        texts = texts.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(texts)

        loss = criterion(outputs, labels)

        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip
            )

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    return avg_loss


# ------------------------------------------------
# Evaluation
# ------------------------------------------------
def evaluate(model, dataloader, criterion, device):

    model.eval()

    total_loss = 0.0

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for texts, labels in dataloader:

            texts = texts.to(device)
            labels = labels.to(device)

            outputs = model(texts)

            loss = criterion(outputs, labels)

            preds = torch.argmax(outputs, dim=1)

            total_loss += loss.item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    accuracy = accuracy_score(all_labels, all_preds)

    precision = precision_score(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0
    )

    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, accuracy, precision, recall, f1, cm


# ------------------------------------------------
# Save learning curves
# ------------------------------------------------
def save_learning_curves(
    train_losses,
    test_losses,
    test_accuracies,
    result_dir
):

    plt.figure()

    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Baseline RNN Classification Loss Curve")

    plt.legend()

    plt.savefig(
        os.path.join(
            result_dir,
            "rnn_classification_loss_curve.png"
        )
    )

    plt.close()

    plt.figure()

    plt.plot(test_accuracies, label="Test Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Baseline RNN Classification Accuracy Curve")

    plt.legend()

    plt.savefig(
        os.path.join(
            result_dir,
            "rnn_classification_accuracy_curve.png"
        )
    )

    plt.close()


# ------------------------------------------------
# Save metrics
# ------------------------------------------------
def save_metrics(
    result_dir,
    vocab_size,
    max_len,
    max_vocab,
    embed_dim,
    hidden_dim,
    batch_size,
    learning_rate,
    num_epochs,
    train_losses,
    test_losses,
    test_accuracies,
    final_accuracy,
    final_precision,
    final_recall,
    final_f1,
    final_cm
):

    metrics_path = os.path.join(
        result_dir,
        "rnn_classification_metrics.txt"
    )

    with open(metrics_path, "w", encoding="utf-8") as f:

        f.write("Baseline RNN Classification Results\n")
        f.write("-----------------------------------\n\n")

        f.write("Dataset Information\n")
        f.write("Dataset: IMDB Sentiment Classification\n")
        f.write("Task: Binary classification, positive vs negative movie reviews\n")
        f.write("Train Samples: 25000\n")
        f.write("Test Samples: 25000\n")
        f.write("Classes: 2\n")
        f.write("Label Mapping: neg = 0, pos = 1\n\n")

        f.write("Hyperparameters\n")
        f.write(f"Vocabulary Size: {vocab_size}\n")
        f.write(f"Max Vocabulary Limit: {max_vocab}\n")
        f.write(f"Max Sequence Length: {max_len}\n")
        f.write(f"Embedding Dimension: {embed_dim}\n")
        f.write(f"Hidden Dimension: {hidden_dim}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Epochs: {num_epochs}\n\n")

        f.write("Final Test Metrics\n")
        f.write(f"Accuracy: {final_accuracy:.4f}\n")
        f.write(f"Precision: {final_precision:.4f}\n")
        f.write(f"Recall: {final_recall:.4f}\n")
        f.write(f"F1 Score: {final_f1:.4f}\n\n")

        f.write("Final Confusion Matrix\n")
        f.write("Rows = true labels, Columns = predicted labels\n")
        f.write("[[TN, FP],\n")
        f.write(" [FN, TP]]\n")
        f.write(str(final_cm))
        f.write("\n\n")

        f.write("Epoch History\n")
        f.write("epoch,train_loss,test_loss,test_accuracy\n")

        for i in range(len(train_losses)):
            f.write(
                f"{i + 1},"
                f"{train_losses[i]:.4f},"
                f"{test_losses[i]:.4f},"
                f"{test_accuracies[i]:.4f}\n"
            )


# ------------------------------------------------
# Main
# ------------------------------------------------
def main():

    set_seed(170)

    classification_dir = os.path.join(
        PROJECT_ROOT,
        "local_code",
        "stage_4_code",
        "stage_4_data",
        "text_classification"
    )

    result_dir = os.path.join(
        PROJECT_ROOT,
        "result",
        "stage_4_result"
    )

    os.makedirs(result_dir, exist_ok=True)

    # ------------------------------------------------
    # Hyperparameters
    # ------------------------------------------------
    max_len = 200
    max_vocab = 10000

    batch_size = 64

    embed_dim = 128
    hidden_dim = 128

    output_dim = 2

    learning_rate = 0.0005

    num_epochs = 10

    grad_clip = 1.0

    # ------------------------------------------------
    # Device
    # ------------------------------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # ------------------------------------------------
    # Load data
    # ------------------------------------------------
    print("Loading data...")

    data, word_to_idx, idx_to_word = load_classification_data(
        classification_dir,
        max_len=max_len,
        max_vocab=max_vocab
    )

    train_dataset = TextClassificationDataset(
        data["train"]["X"],
        data["train"]["y"]
    )

    test_dataset = TextClassificationDataset(
        data["test"]["X"],
        data["test"]["y"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # ------------------------------------------------
    # Build model
    # ------------------------------------------------
    vocab_size = len(word_to_idx)

    model = RNNClassifier(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        pad_idx=PAD_IDX
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    print(model)

    # ------------------------------------------------
    # Training loop
    # ------------------------------------------------
    train_losses = []
    test_losses = []
    test_accuracies = []

    best_accuracy = 0.0
    best_f1 = 0.0

    final_accuracy = 0.0
    final_precision = 0.0
    final_recall = 0.0
    final_f1 = 0.0
    final_cm = None

    best_model_path = os.path.join(
        result_dir,
        "rnn_classifier_best_model.pth"
    )

    for epoch in range(num_epochs):

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip=grad_clip
        )

        test_loss, accuracy, precision, recall, f1, cm = evaluate(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device
        )

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(accuracy)

        final_accuracy = accuracy
        final_precision = precision
        final_recall = recall
        final_f1 = f1
        final_cm = cm

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_f1 = f1

            torch.save(
                model.state_dict(),
                best_model_path
            )

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Best Accuracy So Far: {best_accuracy:.4f}")
        print("-" * 40)

    # ------------------------------------------------
    # Save final model
    # ------------------------------------------------
    final_model_path = os.path.join(
        result_dir,
        "rnn_classifier_final_model.pth"
    )

    torch.save(
        model.state_dict(),
        final_model_path
    )

    # ------------------------------------------------
    # Save results
    # ------------------------------------------------
    save_learning_curves(
        train_losses=train_losses,
        test_losses=test_losses,
        test_accuracies=test_accuracies,
        result_dir=result_dir
    )

    save_metrics(
        result_dir=result_dir,
        vocab_size=vocab_size,
        max_len=max_len,
        max_vocab=max_vocab,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        train_losses=train_losses,
        test_losses=test_losses,
        test_accuracies=test_accuracies,
        final_accuracy=final_accuracy,
        final_precision=final_precision,
        final_recall=final_recall,
        final_f1=final_f1,
        final_cm=final_cm
    )

    print("Training complete.")
    print(f"Final model saved to: {final_model_path}")
    print(f"Best model saved to: {best_model_path}")
    print(f"Results saved to: {result_dir}")


if __name__ == "__main__":
    main()