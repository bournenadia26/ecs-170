#!/usr/bin/env python3
"""
Stage 4 Notebook Generator
===========================
Run this script once to produce:
  stage4_classification.ipynb  (tasks 4-2, 4-3, 4-5)
  stage4_generation.ipynb      (tasks 4-4, 4-5)
"""
import json, uuid, os

def C(s):
    """Code cell."""
    return {"cell_type": "code", "execution_count": None,
            "id": uuid.uuid4().hex[:8], "metadata": {}, "outputs": [],
            "source": s.splitlines(keepends=True)}

def M(s):
    """Markdown cell."""
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8],
            "metadata": {}, "source": s.splitlines(keepends=True)}

def NB(cells):
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {"display_name": "Python 3",
                               "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.9.0",
                                  "file_extension": ".py",
                                  "mimetype": "text/x-python",
                                  "codemirror_mode": {"name": "ipython", "version": 3}}},
            "cells": cells}

BASE = '/Users/kavosh/Desktop/stage4'

# ==============================================================================
#  CLASSIFICATION NOTEBOOK
# ==============================================================================

clf = []

clf.append(M("""\
# Stage 4 – Text Classification with RNN / LSTM / GRU

**Tasks covered**: 4-1 (data exploration) · 4-2 (RNN model) · 4-3 (training + evaluation) · 4-5 (LSTM & GRU)

**Dataset**: IMDB Movie Reviews — 25,000 train / 25,000 test, binary sentiment  
**Models**: Vanilla RNN · LSTM · GRU  

---
Run all cells top-to-bottom. On CPU set `USE_SUBSET = True` (~5 min total).  
On Google Colab with GPU set `USE_SUBSET = False` for full 25 K training.
"""))

clf.append(C("""\
# Uncomment on Google Colab if packages are missing:
# !pip install torch scikit-learn matplotlib seaborn
"""))

clf.append(C("""\
# ── 1. Imports & Device ────────────────────────────────────────────────────────
import os, re, sys, time, random, warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns

warnings.filterwarnings('ignore')

# ── Detect environment ─────────────────────────────────────────────────────────
try:
    import google.colab
    IN_COLAB = True
    print("Running on Google Colab")
except ImportError:
    IN_COLAB = False
    print("Running locally")

# ── Device selection: CUDA > MPS (Apple Silicon) > CPU ────────────────────────
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if device.type == 'cuda':
    torch.cuda.manual_seed_all(SEED)
"""))

clf.append(C("""\
# ── 2. Configuration ───────────────────────────────────────────────────────────
# ┌──────────────────────────────────────────────────────────────────────────────┐
# │  USE_SUBSET = True  → 2500 reviews/class (5 000 total) — fast CPU training  │
# │  USE_SUBSET = False → full 12 500/class (25 000 total) — GPU recommended    │
# └──────────────────────────────────────────────────────────────────────────────┘
USE_SUBSET  = True    # ← flip to False on Colab/GPU
SUBSET_SIZE = 2500    # reviews per class when USE_SUBSET = True

# ── Data paths ─────────────────────────────────────────────────────────────────
if IN_COLAB:
    from google.colab import drive
    import subprocess
    drive.mount('/content/drive')
    # ── Set this to the path of stage_4_data.zip on your Drive ────────────────
    DRIVE_ZIP_PATH = '/content/drive/MyDrive/stage_4_data.zip'  # ← update if yours is in a subfolder
    # ── Unzip once into /content/ (skipped if already done) ───────────────────
    if not os.path.exists('/content/stage_4_data'):
        print(f'Unzipping {DRIVE_ZIP_PATH} ...')
        subprocess.run(['unzip', '-q', DRIVE_ZIP_PATH, '-d', '/content/'], check=True)
        print('Done.')
    BASE_DATA = '/content/stage_4_data'
else:
    BASE_DATA = '/Users/kavosh/Desktop/stage4/stage_4_data/stage_4_data'

TRAIN_DIR = os.path.join(BASE_DATA, 'text_classification', 'train')
TEST_DIR  = os.path.join(BASE_DATA, 'text_classification', 'test')

# ── Model hyper-parameters ─────────────────────────────────────────────────────
VOCAB_SIZE   = 10_000
EMBED_DIM    = 128
HIDDEN_DIM   = 256
N_LAYERS     = 2
DROPOUT      = 0.3
MAX_SEQ_LEN  = 200
BATCH_SIZE   = 64
N_EPOCHS     = 15
LR           = 1e-3

print("Configuration:")
print(f"  USE_SUBSET={USE_SUBSET}, SUBSET_SIZE={SUBSET_SIZE}")
print(f"  VOCAB={VOCAB_SIZE}, EMBED={EMBED_DIM}, HIDDEN={HIDDEN_DIM}")
print(f"  LAYERS={N_LAYERS}, DROPOUT={DROPOUT}, SEQ_LEN={MAX_SEQ_LEN}")
print(f"  BATCH={BATCH_SIZE}, EPOCHS={N_EPOCHS}, LR={LR}")
"""))

clf.append(C("""\
# ── 3. Data Loading & Exploration (Task 4-1) ───────────────────────────────────
def load_split(split_dir, subset_size=None):
    \"\"\"Load reviews from pos/ and neg/ subdirectories.
    Returns (texts, labels) where label 0=negative, 1=positive.
    \"\"\"
    texts, labels = [], []
    for label_idx, cls in enumerate(['neg', 'pos']):
        folder = os.path.join(split_dir, cls)
        fnames = sorted(f for f in os.listdir(folder) if f.endswith('.txt'))
        if subset_size:
            fnames = fnames[:subset_size]
        for fname in fnames:
            with open(os.path.join(folder, fname), 'r',
                      encoding='utf-8', errors='replace') as fh:
                texts.append(fh.read().strip())
            labels.append(label_idx)
    return texts, labels

subset = SUBSET_SIZE if USE_SUBSET else None
train_texts, train_labels = load_split(TRAIN_DIR, subset)
test_texts,  test_labels  = load_split(TEST_DIR,  subset)

print(f"Train: {len(train_texts):,} reviews  "
      f"(neg={train_labels.count(0):,} | pos={train_labels.count(1):,})")
print(f"Test : {len(test_texts):,}  reviews  "
      f"(neg={test_labels.count(0):,} | pos={test_labels.count(1):,})")

# ── Peek at sample reviews ─────────────────────────────────────────────────────
print("\\n── NEGATIVE sample ────────────────────────────────────────────────────")
print(train_texts[0][:400])
print("\\n── POSITIVE sample ────────────────────────────────────────────────────")
pos_idx = next(i for i, l in enumerate(train_labels) if l == 1)
print(train_texts[pos_idx][:400])

# ── Length distribution ────────────────────────────────────────────────────────
lengths = [len(t.split()) for t in train_texts]
print(f"\\nReview lengths: mean={np.mean(lengths):.0f}, "
      f"median={np.median(lengths):.0f}, max={max(lengths)}, min={min(lengths)}")
"""))

clf.append(C("""\
# ── 4. Tokenisation & Vocabulary ───────────────────────────────────────────────
PAD, UNK = '<PAD>', '<UNK>'   # indices 0 and 1

def tokenize(text):
    \"\"\"Lowercase, strip HTML tags, keep alphanumeric only, split on whitespace.\"\"\"
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)           # strip HTML
    text = re.sub(r'[^a-z0-9\\s]', ' ', text)      # keep letters + digits
    return text.split()

def build_vocab(texts, max_size=10_000):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    vocab = {PAD: 0, UNK: 1}
    for word, _ in counter.most_common(max_size - 2):
        vocab[word] = len(vocab)
    return vocab

vocab     = build_vocab(train_texts, VOCAB_SIZE)
idx2word  = {v: k for k, v in vocab.items()}
print(f"Vocabulary size: {len(vocab):,}")
print(f"Top 15 words:    {list(vocab.keys())[2:17]}")
"""))

clf.append(C("""\
# ── 5. Dataset & DataLoader ────────────────────────────────────────────────────
def encode(text, vocab, max_len):
    ids = [vocab.get(t, 1) for t in tokenize(text)[:max_len]]   # 1 = UNK
    ids += [0] * (max_len - len(ids))                            # pad
    return ids

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.X = torch.tensor([encode(t, vocab, max_len) for t in texts],
                              dtype=torch.long)
        self.y = torch.tensor(labels, dtype=torch.float)
    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

pin = device.type == 'cuda'
train_ds = ReviewDataset(train_texts, train_labels, vocab, MAX_SEQ_LEN)
test_ds  = ReviewDataset(test_texts,  test_labels,  vocab, MAX_SEQ_LEN)
train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  pin_memory=pin)
test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, pin_memory=pin)

print(f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")
"""))

clf.append(C("""\
# ── 6. RNN Model Architecture (Task 4-2) ──────────────────────────────────────
class TextRNN(nn.Module):
    \"\"\"
    Unified text classifier supporting RNN / LSTM / GRU.

    Architecture:
        Embedding(vocab, embed_dim)
        → Dropout
        → RNN (unidirectional) / BiLSTM / BiGRU  (n_layers, hidden_dim)
        → Dropout on last hidden state
        → Linear(hidden_dim * [1 or 2], 1)   # *2 for bidirectional LSTM/GRU

    LSTM and GRU use bidirectional=True for better accuracy on long reviews.
    Binary cross-entropy with logits loss is used externally.
    \"\"\"
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers,
                 model_type='RNN', dropout=0.3, pad_idx=0, bidirectional=False):
        super().__init__()
        self.model_type    = model_type
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.drop       = nn.Dropout(dropout)

        rnn_drop = dropout if n_layers > 1 else 0.0   # PyTorch requires 0 for 1-layer
        if model_type == 'RNN':
            self.rnn = nn.RNN(embed_dim, hidden_dim, n_layers,
                              batch_first=True, dropout=rnn_drop, nonlinearity='tanh',
                              bidirectional=bidirectional)
        elif model_type == 'LSTM':
            self.rnn = nn.LSTM(embed_dim, hidden_dim, n_layers,
                               batch_first=True, dropout=rnn_drop,
                               bidirectional=bidirectional)
        elif model_type == 'GRU':
            self.rnn = nn.GRU(embed_dim, hidden_dim, n_layers,
                              batch_first=True, dropout=rnn_drop,
                              bidirectional=bidirectional)
        else:
            raise ValueError(f"Unknown model_type: {model_type!r}")

        fc_in   = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_in, 1)

    def forward(self, x):
        # x : (batch, seq_len)
        emb = self.drop(self.embedding(x))     # (B, T, E)

        if self.model_type == 'LSTM':
            _, (h, _) = self.rnn(emb)          # h : (n_layers * dirs, B, H)
        else:
            _, h = self.rnn(emb)               # h : (n_layers * dirs, B, H)

        if self.bidirectional:
            # Concat last-layer forward + backward hidden states
            out = torch.cat([h[-2], h[-1]], dim=1)   # (B, H*2)
        else:
            out = h[-1]                              # (B, H)

        return self.fc(self.drop(out)).squeeze(-1)   # (B,)


# ── Architecture summary ───────────────────────────────────────────────────────
_demo = TextRNN(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, N_LAYERS, 'GRU', DROPOUT, bidirectional=True)
_x    = torch.randint(0, VOCAB_SIZE, (4, MAX_SEQ_LEN))
assert _demo(_x).shape == (4,), "Shape check failed"
n_params = sum(p.numel() for p in _demo.parameters() if p.requires_grad)
print(_demo)
print(f"\\nTrainable parameters: {n_params:,}")
del _demo, _x
"""))

clf.append(C("""\
# ── 7. Training & Evaluation Utilities ────────────────────────────────────────
criterion = nn.BCEWithLogitsLoss()

def run_epoch(model, loader, optimizer=None, training=True):
    \"\"\"One full pass. Returns (avg_loss, accuracy, preds, labels).\"\"\"
    model.train() if training else model.eval()
    total_loss = total_correct = total_n = 0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss   = criterion(logits, yb)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss    += loss.item() * len(yb)
            preds          = (torch.sigmoid(logits) >= 0.5).long()
            total_correct += (preds == yb.long()).sum().item()
            total_n       += len(yb)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(yb.long().cpu().tolist())

    return total_loss / total_n, total_correct / total_n, all_preds, all_labels


def train_model(model_type):
    \"\"\"Train a TextRNN model and return (model, history).
    LSTM and GRU use bidirectional=True for higher accuracy on long reviews.\"\"\"
    bidir = model_type in ('LSTM', 'GRU')
    model = TextRNN(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, N_LAYERS,
                    model_type, DROPOUT, bidirectional=bidir).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, patience=2, factor=0.5)

    history      = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    best_state   = None

    prefix = 'Bi' if bidir else '  '
    print(f"\\n{'='*62}")
    print(f"  Training  {prefix}{model_type:<6}  |  "
          f"embed={EMBED_DIM}, hidden={HIDDEN_DIM}, layers={N_LAYERS}, bidirectional={bidir}")
    print(f"{'='*62}")
    print(f"{'Ep':>3} | {'Tr Loss':>8} | {'Tr Acc':>7} | "
          f"{'Val Loss':>9} | {'Val Acc':>8} | {'Time':>5}")
    print('-' * 50)

    for ep in range(1, N_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, optimizer)
        va_loss, va_acc, _, _ = run_epoch(model, test_loader,  training=False)
        scheduler.step(va_loss)

        history['train_loss'].append(tr_loss);  history['train_acc'].append(tr_acc)
        history['val_loss'].append(va_loss);    history['val_acc'].append(va_acc)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"{ep:3d} | {tr_loss:8.4f} | {tr_acc:7.4f} | "
              f"{va_loss:9.4f} | {va_acc:8.4f} | {time.time()-t0:4.1f}s")

    model.load_state_dict(best_state)
    print(f"\\n  Best validation accuracy: {best_val_acc:.4f}")
    return model, history
"""))

clf.append(C("""\
# ── 8. Train Vanilla RNN (Task 4-3) ───────────────────────────────────────────
rnn_model, rnn_history = train_model('RNN')
"""))

clf.append(C("""\
# ── 9. Train LSTM (Task 4-5) ──────────────────────────────────────────────────
lstm_model, lstm_history = train_model('LSTM')
"""))

clf.append(C("""\
# ── 10. Train GRU (Task 4-5) ──────────────────────────────────────────────────
gru_model, gru_history = train_model('GRU')
"""))

clf.append(C("""\
# ── 11. Learning Curves (Task 4-3 / 4-5) ─────────────────────────────────────
histories = {'RNN': rnn_history, 'LSTM': lstm_history, 'GRU': gru_history}
palette   = {'RNN': '#e74c3c', 'LSTM': '#2ecc71', 'GRU': '#3498db'}
epochs    = range(1, N_EPOCHS + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Learning Curves – Text Classification', fontsize=14, fontweight='bold')

for name, hist in histories.items():
    c = palette[name]
    ax1.plot(epochs, hist['train_loss'], '--', color=c, alpha=0.55, label=f'{name} train')
    ax1.plot(epochs, hist['val_loss'],   '-',  color=c, lw=2,       label=f'{name} val')
    ax2.plot(epochs, hist['train_acc'],  '--', color=c, alpha=0.55, label=f'{name} train')
    ax2.plot(epochs, hist['val_acc'],    '-',  color=c, lw=2,       label=f'{name} val')

for ax, title, ylabel in [(ax1, 'Loss', 'BCE Loss'), (ax2, 'Accuracy', 'Accuracy')]:
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Epoch');  ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, ncol=2);  ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath('__file__')),
            'classification_learning_curves.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: classification_learning_curves.png")
"""))

clf.append(C("""\
# ── 12. Final Evaluation on Test Set (Task 4-3 / 4-5) ────────────────────────
print("\\n" + "="*62)
print("  EVALUATION RESULTS ON TEST SET")
print("="*62)

results = {}
for name, model in [('RNN', rnn_model), ('LSTM', lstm_model), ('GRU', gru_model)]:
    _, acc, preds, labels = run_epoch(model, test_loader, training=False)
    f1  = f1_score(labels, preds, average='binary')
    results[name] = dict(accuracy=acc, f1=f1, preds=preds, labels=labels)
    print(f"\\n─── {name} ──────────────────────────────────────────────────────")
    print(f"  Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  F1 Score : {f1:.4f}")
    print(classification_report(labels, preds,
                                 target_names=['Negative', 'Positive'], digits=4))
"""))

clf.append(C("""\
# ── 13. Confusion Matrices ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Confusion Matrices – Test Set', fontsize=14, fontweight='bold')

for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(res['labels'], res['preds'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'], cbar=False)
    ax.set_title(f'{name}  (Acc={res["accuracy"]:.4f})', fontsize=11)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath('__file__')),
            'classification_confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: classification_confusion_matrices.png")
"""))

clf.append(C("""\
# ── 14. Summary Table ─────────────────────────────────────────────────────────
print("\\n" + "="*42)
print(f"{'Model':<8} | {'Accuracy':>10} | {'F1 Score':>10}")
print("-" * 42)
for name, res in results.items():
    print(f"{name:<8} | {res['accuracy']:10.4f} | {res['f1']:10.4f}")
print("="*42)

best_model = max(results, key=lambda k: results[k]['accuracy'])
print(f"\\nBest model: {best_model} "
      f"(accuracy={results[best_model]['accuracy']:.4f})")
"""))

# ==============================================================================
#  GENERATION NOTEBOOK
# ==============================================================================

gen = []

gen.append(M("""\
# Stage 4 – Text Generation with RNN / LSTM / GRU

**Tasks covered**: 4-1 (data exploration) · 4-4 (training + generation) · 4-5 (LSTM & GRU)

**Dataset**: Short Jokes — 1,623 one-liner jokes  
**Models**: Vanilla RNN · LSTM · GRU  
**Task**: Train a word-level language model; generate jokes starting with 3 seed words.

---
Architecture: `Embedding → RNN/LSTM/GRU → Linear(vocab_size)`  
Loss: `CrossEntropyLoss` (next-word prediction)
"""))

gen.append(C("""\
# Uncomment on Google Colab if packages are missing:
# !pip install torch scikit-learn matplotlib
"""))

gen.append(C("""\
# ── 1. Imports & Device ────────────────────────────────────────────────────────
import os, re, csv, sys, time, random, math, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

try:
    import google.colab; IN_COLAB = True; print("Running on Google Colab")
except ImportError:
    IN_COLAB = False; print("Running locally")

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if device.type == 'cuda':
    torch.cuda.manual_seed_all(SEED)
"""))

gen.append(C("""\
# ── 2. Configuration ───────────────────────────────────────────────────────────
if IN_COLAB:
    from google.colab import drive
    import subprocess
    drive.mount('/content/drive')
    # ── Set this to the path of stage_4_data.zip on your Drive ────────────────
    DRIVE_ZIP_PATH = '/content/drive/MyDrive/stage_4_data.zip'  # ← update if yours is in a subfolder
    # ── Unzip once into /content/ (skipped if already done) ───────────────────
    if not os.path.exists('/content/stage_4_data'):
        print(f'Unzipping {DRIVE_ZIP_PATH} ...')
        subprocess.run(['unzip', '-q', DRIVE_ZIP_PATH, '-d', '/content/'], check=True)
        print('Done.')
    DATA_FILE = '/content/stage_4_data/text_generation/data'
else:
    DATA_FILE = '/Users/kavosh/Desktop/stage4/stage_4_data/stage_4_data/text_generation/data'

# ── Hyper-parameters ───────────────────────────────────────────────────────────
VOCAB_SIZE  = 5_000    # most common words kept
SEQ_LEN     = 30       # input sequence length (tokens)
EMBED_DIM   = 128
HIDDEN_DIM  = 256
N_LAYERS    = 2
DROPOUT     = 0.3
BATCH_SIZE  = 64
N_EPOCHS    = 20       # more epochs needed for generation
LR          = 1e-3
TEMPERATURE = 0.8      # sampling temperature (lower = more conservative)
MAX_GEN_LEN = 50       # max tokens to generate

PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIAL = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']

print(f"VOCAB={VOCAB_SIZE}, SEQ_LEN={SEQ_LEN}, "
      f"EMBED={EMBED_DIM}, HIDDEN={HIDDEN_DIM}")
"""))

gen.append(C("""\
# ── 3. Data Loading & Exploration (Task 4-1) ───────────────────────────────────
def load_jokes(filepath):
    \"\"\"Parse the CSV file (ID, Joke columns) and return list of joke strings.\"\"\"
    jokes = []
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            joke = row.get('Joke', '').strip()
            if joke:
                jokes.append(joke)
    return jokes

jokes = load_jokes(DATA_FILE)
print(f"Total jokes: {len(jokes)}")
print("\\n── Sample jokes ──────────────────────────────────────────────────────────")
for i in [0, 1, 2, 100, 500]:
    print(f"  [{i}] {jokes[i][:120]}")

# ── Length distribution ────────────────────────────────────────────────────────
lengths = [len(j.split()) for j in jokes]
print(f"\\nJoke word counts: mean={np.mean(lengths):.1f}, "
      f"median={np.median(lengths):.1f}, max={max(lengths)}, min={min(lengths)}")

plt.figure(figsize=(8, 3))
plt.hist(lengths, bins=40, color='steelblue', edgecolor='white')
plt.title('Joke Length Distribution (words)'); plt.xlabel('# Words'); plt.ylabel('Count')
plt.tight_layout(); plt.show()
"""))

gen.append(C("""\
# ── 4. Tokenisation & Vocabulary ───────────────────────────────────────────────
def tokenize_joke(text):
    text = text.lower()
    text = re.sub(r\"[^a-z0-9'\\s]\", ' ', text)   # keep letters, digits, apostrophe
    return text.split()

def build_vocab(jokes, max_size):
    counter = Counter()
    for j in jokes:
        counter.update(tokenize_joke(j))
    vocab = {s: i for i, s in enumerate(SPECIAL)}
    for word, _ in counter.most_common(max_size - len(SPECIAL)):
        vocab[word] = len(vocab)
    return vocab

vocab    = build_vocab(jokes, VOCAB_SIZE)
idx2word = {v: k for k, v in vocab.items()}
actual_vocab = len(vocab)
print(f"Actual vocabulary size: {actual_vocab:,}  (capped at {VOCAB_SIZE:,})")
print(f"Top-20 content words: "
      f"{[w for w in list(vocab.keys())[4:24]]}")
"""))

gen.append(C("""\
# ── 5. Joke Dataset (sliding-window next-word prediction) ─────────────────────
def encode_joke(joke, vocab, seq_len):
    \"\"\"
    Returns list of (input_ids, target_ids) training windows from one joke.
    Each window: input = seq_len tokens, target = same tokens shifted by 1.
    \"\"\"
    tokens = [SOS_IDX] + [vocab.get(t, UNK_IDX) for t in tokenize_joke(joke)] + [EOS_IDX]
    pairs  = []
    for start in range(0, len(tokens) - seq_len):
        inp = tokens[start : start + seq_len]
        tgt = tokens[start + 1 : start + seq_len + 1]
        pairs.append((inp, tgt))
    return pairs

class JokeDataset(Dataset):
    def __init__(self, jokes, vocab, seq_len):
        self.pairs = []
        for j in jokes:
            self.pairs.extend(encode_joke(j, vocab, seq_len))
    def __len__(self):  return len(self.pairs)
    def __getitem__(self, i):
        inp, tgt = self.pairs[i]
        return (torch.tensor(inp, dtype=torch.long),
                torch.tensor(tgt, dtype=torch.long))

# ── Split 90% train / 10% validation ──────────────────────────────────────────
random.shuffle(jokes)
split     = int(0.9 * len(jokes))
train_j, val_j = jokes[:split], jokes[split:]

train_ds  = JokeDataset(train_j, vocab, SEQ_LEN)
val_ds    = JokeDataset(val_j,   vocab, SEQ_LEN)
pin       = device.type == 'cuda'
train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  pin_memory=pin)
val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, pin_memory=pin)

print(f"Training windows: {len(train_ds):,} | Validation windows: {len(val_ds):,}")
print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
"""))

gen.append(C("""\
# ── 6. Language Model Architecture (Task 4-2 / 4-4) ──────────────────────────
class JokeLM(nn.Module):
    \"\"\"
    Word-level language model.
    Architecture:
        Embedding(vocab, embed_dim)
        → Dropout
        → RNN / LSTM / GRU
        → Dropout
        → Linear(hidden_dim, vocab_size)

    forward() returns (logits, hidden):
        logits : (B, T, vocab_size)  — raw scores for next-word prediction
        hidden : last RNN hidden state (for stateful generation)
    \"\"\"
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers,
                 model_type='RNN', dropout=0.3):
        super().__init__()
        self.model_type = model_type
        self.hidden_dim  = hidden_dim
        self.n_layers    = n_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.drop       = nn.Dropout(dropout)

        rnn_drop = dropout if n_layers > 1 else 0.0
        if model_type == 'RNN':
            self.rnn = nn.RNN(embed_dim, hidden_dim, n_layers,
                              batch_first=True, dropout=rnn_drop)
        elif model_type == 'LSTM':
            self.rnn = nn.LSTM(embed_dim, hidden_dim, n_layers,
                               batch_first=True, dropout=rnn_drop)
        elif model_type == 'GRU':
            self.rnn = nn.GRU(embed_dim, hidden_dim, n_layers,
                              batch_first=True, dropout=rnn_drop)
        else:
            raise ValueError(f"Unknown model_type: {model_type!r}")

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        emb    = self.drop(self.embedding(x))    # (B, T, E)
        out, h = self.rnn(emb, hidden)           # out: (B, T, H)
        logits = self.fc(self.drop(out))         # (B, T, V)
        return logits, h

    def init_hidden(self, batch_size):
        z = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        return (z, z.clone()) if self.model_type == 'LSTM' else z


# ── Sanity check ──────────────────────────────────────────────────────────────
_m  = JokeLM(actual_vocab, EMBED_DIM, HIDDEN_DIM, N_LAYERS, 'RNN', DROPOUT)
_x  = torch.randint(0, actual_vocab, (4, SEQ_LEN))
_lo, _ = _m(_x)
assert _lo.shape == (4, SEQ_LEN, actual_vocab), "Shape check failed"
n_params = sum(p.numel() for p in _m.parameters() if p.requires_grad)
print(_m)
print(f"\\nTrainable parameters: {n_params:,}")
del _m, _x, _lo
"""))

gen.append(C("""\
# ── 7. Training & Evaluation Utilities ────────────────────────────────────────
gen_criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

def run_gen_epoch(model, loader, optimizer=None, training=True):
    \"\"\"One epoch. Returns (avg_loss, perplexity).\"\"\"
    model.train() if training else model.eval()
    total_loss = total_tokens = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            hidden  = model.init_hidden(xb.size(0))
            logits, _ = model(xb, hidden)
            # logits: (B, T, V); yb: (B, T)  → reshape for loss
            loss    = gen_criterion(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            n_tok        = (yb != PAD_IDX).sum().item()
            total_loss  += loss.item() * n_tok
            total_tokens += n_tok

    avg_loss = total_loss / max(total_tokens, 1)
    perp     = math.exp(min(avg_loss, 100))
    return avg_loss, perp


def train_gen_model(model_type):
    \"\"\"Train JokeLM and return (model, history).\"\"\"
    model     = JokeLM(actual_vocab, EMBED_DIM, HIDDEN_DIM, N_LAYERS,
                       model_type, DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, patience=3, factor=0.5)

    history      = {'train_loss': [], 'val_loss': [], 'train_ppl': [], 'val_ppl': []}
    best_val_loss = float('inf')
    best_state    = None

    print(f"\\n{'='*62}")
    print(f"  Training  {model_type:<6}  |  "
          f"embed={EMBED_DIM}, hidden={HIDDEN_DIM}, layers={N_LAYERS}")
    print(f"{'='*62}")
    print(f"{'Ep':>3} | {'Tr Loss':>8} | {'Tr PPL':>8} | "
          f"{'Val Loss':>9} | {'Val PPL':>8} | {'Time':>5}")
    print('-' * 55)

    for ep in range(1, N_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_ppl = run_gen_epoch(model, train_loader, optimizer)
        va_loss, va_ppl = run_gen_epoch(model, val_loader,   training=False)
        scheduler.step(va_loss)

        history['train_loss'].append(tr_loss);  history['train_ppl'].append(tr_ppl)
        history['val_loss'].append(va_loss);    history['val_ppl'].append(va_ppl)

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"{ep:3d} | {tr_loss:8.4f} | {tr_ppl:8.2f} | "
              f"{va_loss:9.4f} | {va_ppl:8.2f} | {time.time()-t0:4.1f}s")

    model.load_state_dict(best_state)
    print(f"\\n  Best val loss: {best_val_loss:.4f}  "
          f"(perplexity={math.exp(min(best_val_loss, 100)):.2f})")
    return model, history
"""))

gen.append(C("""\
# ── 8. Train Vanilla RNN (Task 4-4) ───────────────────────────────────────────
rnn_gen,  rnn_gen_hist  = train_gen_model('RNN')
"""))

gen.append(C("""\
# ── 9. Train LSTM (Task 4-5) ──────────────────────────────────────────────────
lstm_gen, lstm_gen_hist = train_gen_model('LSTM')
"""))

gen.append(C("""\
# ── 10. Train GRU (Task 4-5) ──────────────────────────────────────────────────
gru_gen,  gru_gen_hist  = train_gen_model('GRU')
"""))

gen.append(C("""\
# ── 11. Text Generation Function (Task 4-4) ───────────────────────────────────
def generate_text(model, seed_words, vocab, idx2word,
                  max_len=MAX_GEN_LEN, temperature=TEMPERATURE, top_k=10):
    \"\"\"
    Generate text from a trained language model.

    Args:
        seed_words : list of 3+ strings to start generation from
        temperature: >1 → more random; <1 → more conservative
        top_k      : sample from top-k predictions at each step
    Returns:
        generated string
    \"\"\"
    model.eval()
    tokens = [SOS_IDX] + [vocab.get(w.lower(), UNK_IDX) for w in seed_words]
    tokens_t = torch.tensor([tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        logits, hidden = model(tokens_t)
        # Take hidden state after processing seed
        generated = list(seed_words)

        for _ in range(max_len):
            # Feed only the last token
            last = torch.tensor([[tokens[-1]]], dtype=torch.long, device=device)
            logits, hidden = model(last, hidden)
            logits = logits[:, -1, :] / temperature   # (1, V)

            # Top-k sampling
            if top_k > 0:
                top_vals, top_idx = torch.topk(logits, k=min(top_k, logits.size(-1)))
                probs = F.softmax(top_vals, dim=-1)
                chosen = top_idx[0, torch.multinomial(probs[0], 1).item()].item()
            else:
                probs  = F.softmax(logits, dim=-1)
                chosen = torch.multinomial(probs[0], 1).item()

            if chosen == EOS_IDX:
                break
            if chosen not in (PAD_IDX, UNK_IDX, SOS_IDX):
                generated.append(idx2word.get(chosen, '<UNK>'))
            tokens.append(chosen)

    return ' '.join(generated)


# ── Test generation quickly ────────────────────────────────────────────────────
seed = ['what', 'did', 'the']
print("Quick generation test (LSTM):")
print(" ", generate_text(lstm_gen, seed, vocab, idx2word))
"""))

gen.append(C("""\
# ── 12. Generate Jokes with All Three Models (Task 4-4 / 4-5) ────────────────
SEED_SETS = [
    ['what', 'did', 'the'],
    ['why', 'did', 'the'],
    ['i',   'told', 'my'],
    ['how', 'many', 'people'],
]

print("=" * 70)
print("  GENERATED JOKES (seed = first 3 words)")
print("=" * 70)

gen_results = {}
for model_type, model in [('RNN', rnn_gen), ('LSTM', lstm_gen), ('GRU', gru_gen)]:
    gen_results[model_type] = {}
    print(f"\\n─── {model_type} ────────────────────────────────────────────────────────")
    for seeds in SEED_SETS:
        text = generate_text(model, seeds, vocab, idx2word,
                             temperature=TEMPERATURE, top_k=10)
        key  = ' '.join(seeds)
        gen_results[model_type][key] = text
        print(f"  Seed: '{key}'")
        print(f"  ▶ {text}\\n")
"""))

gen.append(C("""\
# ── 13. Compare Generated vs Training Data (Task 4-4) ────────────────────────
print("=" * 70)
print("  COMPARISON: Generated Jokes vs Training Examples")
print("=" * 70)

for seeds in SEED_SETS[:2]:
    seed_str = ' '.join(seeds)
    print(f"\\nSeed: '{seed_str}'")
    print("─" * 50)

    # Find real jokes in training data that start with these words
    pattern = seed_str.lower()
    matches = [j for j in train_j if j.lower().startswith(pattern)][:3]
    if not matches:
        matches = [j for j in jokes if any(seed_str.lower() in j.lower()
                                           for _ in [1])][:3]

    if matches:
        print(f"  Real training jokes (containing '{seed_str}'):")
        for m in matches[:2]:
            print(f"    ✓ {m[:120]}")
    else:
        print(f"  (No training jokes found starting with '{seed_str}')")

    print(f"  Generated jokes:")
    for mtype in ['RNN', 'LSTM', 'GRU']:
        txt = gen_results[mtype].get(seed_str, '')
        print(f"    [{mtype}] {txt[:120]}")
"""))

gen.append(C("""\
# ── 14. Learning Curves & Perplexity ─────────────────────────────────────────
histories = {'RNN': rnn_gen_hist, 'LSTM': lstm_gen_hist, 'GRU': gru_gen_hist}
palette   = {'RNN': '#e74c3c', 'LSTM': '#2ecc71', 'GRU': '#3498db'}
epochs    = range(1, N_EPOCHS + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Learning Curves – Text Generation (Language Model)',
             fontsize=14, fontweight='bold')

for name, hist in histories.items():
    c = palette[name]
    ax1.plot(epochs, hist['train_loss'], '--', color=c, alpha=0.55, label=f'{name} train')
    ax1.plot(epochs, hist['val_loss'],   '-',  color=c, lw=2,       label=f'{name} val')
    ax2.plot(epochs, hist['train_ppl'],  '--', color=c, alpha=0.55, label=f'{name} train')
    ax2.plot(epochs, hist['val_ppl'],    '-',  color=c, lw=2,       label=f'{name} val')

ax1.set_title('Cross-Entropy Loss'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax2.set_title('Perplexity');         ax2.set_xlabel('Epoch'); ax2.set_ylabel('Perplexity')
for ax in (ax1, ax2):
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath('__file__')),
            'generation_learning_curves.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: generation_learning_curves.png")
"""))

gen.append(C("""\
# ── 15. Summary ──────────────────────────────────────────────────────────────
import math
print("\\n" + "="*50)
print(f"{'Model':<8} | {'Final Val Loss':>14} | {'Perplexity':>11}")
print("-" * 50)
for name, hist in histories.items():
    loss = hist['val_loss'][-1]
    ppl  = math.exp(min(loss, 100))
    print(f"{name:<8} | {loss:14.4f} | {ppl:11.2f}")
print("="*50)

print("\\nGeneration quality notes:")
print("  • Lower perplexity → model is more certain about next-word predictions")
print("  • LSTM/GRU typically outperform vanilla RNN due to gating mechanisms")
print("  • Evaluate generated jokes for: coherence, grammar, humor structure")
"""))

# ==============================================================================
#  Save both notebooks
# ==============================================================================
clf_nb = NB(clf)
gen_nb = NB(gen)

with open(os.path.join(BASE, 'stage4_classification.ipynb'), 'w', encoding='utf-8') as f:
    json.dump(clf_nb, f, indent=1, ensure_ascii=False)
print(f"Saved: {os.path.join(BASE, 'stage4_classification.ipynb')}")

with open(os.path.join(BASE, 'stage4_generation.ipynb'), 'w', encoding='utf-8') as f:
    json.dump(gen_nb, f, indent=1, ensure_ascii=False)
print(f"Saved: {os.path.join(BASE, 'stage4_generation.ipynb')}")
print("Done.")
