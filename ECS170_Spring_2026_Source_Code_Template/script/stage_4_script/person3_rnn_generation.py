"""
Person 3: Baseline RNN for Text Generation
Dataset: jokes dataset (text_generation/data)
Task:
  - Implement RNN text generation model
  - Train on the jokes dataset
  - Generate text starting from 3 given words
  - Save learning curves and generated examples
"""

import csv
import os
import re
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "../../")
)
DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "local_code",
    "stage_4_code",
    "stage_4_data",
    "text_generation",
    "data"
)
OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "result",
    "stage_4_result"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
SEQ_LEN    = 10          # input sequence length (n-gram window)
EMBED_DIM  = 64
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT    = 0.3
BATCH_SIZE = 64
EPOCHS     = 30
LR         = 0.001
MIN_FREQ   = 2           # minimum word frequency to enter vocab

# Special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

# ── 1. Load and clean data ────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Lowercase, remove punctuation (keep apostrophes), collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_jokes(path: str):
    jokes = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)                        # skip header
        for row in reader:
            if len(row) >= 2 and row[1].strip():
                jokes.append(clean_text(row[1]))
    return jokes


jokes = load_jokes(DATA_PATH)
print(f"Loaded {len(jokes)} jokes")
print(f"Sample: {jokes[0]}")

# ── 2. Build vocabulary ───────────────────────────────────────────────────────

from collections import Counter

all_tokens = [tok for joke in jokes for tok in joke.split()]
freq       = Counter(all_tokens)
vocab_words = [w for w, c in freq.items() if c >= MIN_FREQ]
vocab       = [PAD_TOKEN, UNK_TOKEN] + sorted(vocab_words)

word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
VOCAB_SIZE = len(vocab)
PAD_IDX    = word2idx[PAD_TOKEN]
UNK_IDX    = word2idx[UNK_TOKEN]

print(f"Vocabulary size: {VOCAB_SIZE}")


def encode(tokens):
    return [word2idx.get(t, UNK_IDX) for t in tokens]


# ── 3. PyTorch Dataset ────────────────────────────────────────────────────────

class JokeDataset(Dataset):
    """Sliding-window next-word prediction dataset."""
    def __init__(self, jokes, seq_len):
        self.samples = []
        for joke in jokes:
            tokens = joke.split()
            ids    = encode(tokens)
            # create (input_seq, target) pairs with a sliding window
            for i in range(len(ids) - seq_len):
                x = ids[i : i + seq_len]
                y = ids[i + seq_len]
                self.samples.append((torch.tensor(x, dtype=torch.long),
                                     torch.tensor(y, dtype=torch.long)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


dataset    = JokeDataset(jokes, SEQ_LEN)
train_size = int(0.9 * len(dataset))
val_size   = len(dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(
    dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

print(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")

# ── 4. Model ──────────────────────────────────────────────────────────────────

class RNNTextGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim,
                 num_layers, dropout, cell_type="RNN"):
        super().__init__()
        self.cell_type  = cell_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim,
                                      padding_idx=PAD_IDX)
        rnn_cls = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[cell_type]
        self.rnn = rnn_cls(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))          # (B, T, E)
        out, _ = self.rnn(emb)                         # (B, T, H)
        last    = out[:, -1, :]                        # (B, H)
        logits  = self.fc(self.dropout(last))          # (B, V)
        return logits


# ── 5. Training utilities ─────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y   = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def train_model(cell_type="RNN"):
    model     = RNNTextGenerator(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM,
                                 NUM_LAYERS, DROPOUT, cell_type).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5)

    train_losses, val_losses = [], []
    best_val = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        vl_loss = evaluate(model, val_loader, criterion)
        scheduler.step(vl_loss)
        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        if vl_loss < best_val:
            best_val   = vl_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"  [{cell_type}] Epoch {epoch:3d}/{EPOCHS} "
                  f"| Train Loss {tr_loss:.4f} | Val Loss {vl_loss:.4f}")

    model.load_state_dict(best_state)
    return model, train_losses, val_losses


# ── 6. Text generation ────────────────────────────────────────────────────────

def generate_text(model, start_words: list, max_len: int = 30,
                  temperature: float = 0.8) -> str:
    """
    Generate text starting from `start_words`.
    Uses temperature sampling: lower = more greedy, higher = more random.
    """
    model.eval()
    tokens = [clean_text(w) for w in start_words]
    ids    = encode(tokens)

    # pad or truncate to SEQ_LEN
    if len(ids) < SEQ_LEN:
        ids = [PAD_IDX] * (SEQ_LEN - len(ids)) + ids
    else:
        ids = ids[-SEQ_LEN:]

    generated = list(tokens)

    with torch.no_grad():
        for _ in range(max_len):
            x      = torch.tensor([ids], dtype=torch.long).to(device)
            logits = model(x)                              # (1, V)
            logits = logits / temperature
            probs  = torch.softmax(logits, dim=-1).squeeze(0)
            next_id = torch.multinomial(probs, 1).item()
            next_w  = idx2word[next_id]
            if next_w in (PAD_TOKEN, UNK_TOKEN):
                break
            generated.append(next_w)
            ids = ids[1:] + [next_id]

    return " ".join(generated)


# ── 7. Plot learning curves ───────────────────────────────────────────────────

def plot_curves(results: dict, filename: str):
    """
    results = { cell_type: (train_losses, val_losses), ... }
    """
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4),
                             sharey=False)
    if len(results) == 1:
        axes = [axes]

    for ax, (cell_type, (tr, vl)) in zip(axes, results.items()):
        epochs = range(1, len(tr) + 1)
        ax.plot(epochs, tr, label="Train Loss",      color="steelblue")
        ax.plot(epochs, vl, label="Validation Loss", color="tomato", linestyle="--")
        ax.set_title(f"{cell_type} – Learning Curves")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")


# ── 8. Main experiment ────────────────────────────────────────────────────────

# Three sets of starting words (each exactly 3 words)
START_WORDS = [
    ["why", "did", "the"],
    ["what", "do", "you"],
    ["i", "told", "my"],
]

# For Person 3 baseline we only train the vanilla RNN
# (Person 5 will add LSTM / GRU comparisons)

results   = {}
generated = {}

print("\n" + "="*60)
print("Training baseline RNN model …")
print("="*60)

model_rnn, tr_rnn, vl_rnn = train_model("RNN")
results["RNN"] = (tr_rnn, vl_rnn)

# Generate text for each set of starting words
generated["RNN"] = []
for sw in START_WORDS:
    text = generate_text(model_rnn, sw, max_len=30, temperature=0.8)
    generated["RNN"].append({"start": sw, "generated": text})
    print(f"\n  Start: {sw}")
    print(f"  Generated: {text}")

# ── 9. Save learning curves ───────────────────────────────────────────────────

plot_curves(results, os.path.join(OUTPUT_DIR, "rnn_generation_curves.png"))

# ── 10. Save generated text to file ──────────────────────────────────────────

report_path = os.path.join(OUTPUT_DIR, "generated_texts.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("Person 3 – Baseline RNN Text Generation Results\n")
    f.write("="*60 + "\n\n")
    for cell_type, samples in generated.items():
        f.write(f"Model: {cell_type}\n")
        f.write("-"*40 + "\n")
        for s in samples:
            f.write(f"Starting words : {' '.join(s['start'])}\n")
            f.write(f"Generated text : {s['generated']}\n\n")
        final_tr  = results[cell_type][0][-1]
        final_val = results[cell_type][1][-1]
        best_val  = min(results[cell_type][1])
        f.write(f"Final train loss : {final_tr:.4f}\n")
        f.write(f"Final val loss   : {final_val:.4f}\n")
        f.write(f"Best val loss    : {best_val:.4f}\n\n")

print(f"\nSaved generated texts → {report_path}")

# ── 11. Save the model ────────────────────────────────────────────────────────

model_path = os.path.join(OUTPUT_DIR, "rnn_generation_model.pt")
torch.save({
    "model_state_dict": model_rnn.state_dict(),
    "word2idx": word2idx,
    "idx2word": idx2word,
    "config": {
        "vocab_size": VOCAB_SIZE,
        "embed_dim":  EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "dropout":    DROPOUT,
        "cell_type":  "RNN",
        "seq_len":    SEQ_LEN,
    }
}, model_path)
print(f"Saved model → {model_path}")

# ── 12. Quick comparison: training data vs generated ─────────────────────────

print("\n" + "="*60)
print("Comparison: Generated text vs Training data")
print("="*60)

for cell_type, samples in generated.items():
    print(f"\n[{cell_type}]")
    for s in samples:
        start_str = " ".join(s["start"])
        gen_words = s["generated"].split()
        # find jokes in training set that contain the starting words
        matches = [j for j in jokes if start_str in j]
        print(f"  Start       : \"{start_str}\"")
        print(f"  Generated   : \"{s['generated']}\"")
        print(f"  Train match : {matches[0] if matches else 'No exact match found'}")
        print()

print("\nDone! All results saved to:", OUTPUT_DIR)
