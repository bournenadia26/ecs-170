# Sentiment Analysis & Text Generation with RNN / LSTM / GRU

**ECS 170 — Introduction to Artificial Intelligence, UC Davis (Stage 4)**  
Author: Kavosh Hosseini · SID: 924034149 · skhosseini@ucdavis.edu

---

## Overview

This project implements and compares three recurrent architectures — **Vanilla RNN**, **LSTM**, and **GRU** — across two NLP tasks:

| Task | Dataset | Best Result |
|------|---------|-------------|
| Sentiment Classification | IMDB (25K train / 25K test) | **BiGRU — 87.55% accuracy** |
| Text Generation | Short Jokes CSV (1,622 jokes) | **RNN — 983.35 perplexity** |

---

## Tasks

### Task 1 — Text Classification (IMDB Sentiment)
Binary sentiment classification (positive / negative) on IMDB movie reviews.  
LSTM and GRU use **bidirectional processing** for better long-range context.

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| RNN | 50.49% | 0.5059 | 0.5049 | 0.4830 |
| BiLSTM | 87.02% | 0.8706 | 0.8702 | 0.8701 |
| **BiGRU** | **87.55%** | **0.8755** | **0.8755** | **0.8755** |

> Vanilla RNN fails on long reviews (~234 words avg) due to vanishing gradients.  
> BiLSTM/BiGRU both exceed the 85% accuracy requirement.

### Task 2 — Text Generation (Short Jokes)
Next-word language modelling on a dataset of 1,622 short jokes.

| Model | Best Val Loss | Perplexity |
|-------|--------------|------------|
| **RNN** | **6.8910** | **983.35** |
| LSTM | 7.0394 | 1140.68 |
| GRU | 6.9950 | 1091.12 |

> RNN outperforms gated models here because jokes are short (avg 14.7 words) —  
> no long-range dependency problem, and the simpler model generalises better on the tiny dataset.

---

## Project Structure

```
├── stage4_classification.ipynb   # Classification notebook (Tasks 4-2, 4-3, 4-5)
├── stage4_generation.ipynb       # Generation notebook (Tasks 4-4, 4-5)
├── generate_notebooks.py         # Script that generates both .ipynb files
├── fill_report.py                # Script that fills the report template
├── Stage_4_Report_Filled.docx    # Final report
├── classification_learning_curves.png
├── classification_confusion_matrices.png
└── generation_learning_curves.png
```

---

## Setup & Usage

### Run on Google Colab (recommended — free T4 GPU)

1. Upload `stage4_classification.ipynb` or `stage4_generation.ipynb` to Colab
2. Upload `stage_4_data.zip` to Google Drive
3. In the **Configuration** cell set:
   ```python
   DRIVE_ZIP_PATH = '/content/drive/MyDrive/stage_4_data.zip'
   USE_SUBSET = False   # use full dataset for best accuracy
   ```
4. Run all cells

### Regenerate notebooks locally

```bash
python3 generate_notebooks.py
```

### Regenerate report (after placing PNGs in this folder)

```bash
python3 fill_report.py
```

---

## Model Architecture

### Classification — `TextRNN`
```
Embedding(10000, 128)
→ Dropout(0.3)
→ RNN / BiLSTM / BiGRU  (2 layers, hidden=256)
→ Dropout(0.3) on last hidden state
→ Linear(256 [or 512 for bidirectional], 1)
→ BCEWithLogitsLoss
```

### Generation — `JokeLM`
```
Embedding(vocab, 128)
→ Dropout(0.3)
→ RNN / LSTM / GRU  (2 layers, hidden=256)
→ Dropout(0.3)
→ Linear(256, vocab_size)
→ CrossEntropyLoss
```

---

## Requirements

```
torch
scikit-learn
matplotlib
seaborn
python-docx
```

Install via: `pip install torch scikit-learn matplotlib seaborn python-docx`
