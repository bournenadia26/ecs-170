'''
Dataset Loader for Stage 4: Text Classification and Generation
Handles:
  - IMDB sentiment reviews (classification)
  - Short jokes (generation)

Output format mirrors the CNN/MLP pattern:
    data = {
        'train': {'X': list_of_index_sequences, 'y': list_of_labels},
        'test':  {'X': list_of_index_sequences, 'y': list_of_labels},
    }

For generation there are no labels, so 'y' contains the target sequences
(input sequence shifted by one position — next-word prediction style).
'''

import os
import random
import re
import string
from collections import Counter

# ---------------------------------------------------------------------------
# Special tokens
# Every vocabulary gets these four reserved slots at the front.
# ---------------------------------------------------------------------------
PAD_TOKEN = '<PAD>'   # used to pad short sequences to max_len
UNK_TOKEN = '<UNK>'   # replaces words not seen in the vocabulary
SOS_TOKEN = '<SOS>'   # "start of sequence" — used for generation
EOS_TOKEN = '<EOS>'   # "end of sequence"   — used for generation

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]

PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
def clean_text(text):
    '''
    Lowercase, remove HTML tags, strip punctuation, collapse whitespace.
    Returns a plain string ready for tokenization.
    '''
    # lowercase everything
    text = text.lower()
    # remove HTML tags like <br />, common in IMDB reviews
    text = re.sub(r'<[^>]+>', ' ', text)
    # remove punctuation (keeps only letters, digits, spaces)
    text = text.translate(str.maketrans('', '', string.punctuation))
    # collapse multiple spaces into one
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text):
    '''Split a cleaned string into a list of word tokens.'''
    return text.split()


# ---------------------------------------------------------------------------
# Vocabulary builder
# ---------------------------------------------------------------------------
def build_vocab(token_lists, max_vocab=10000):
    '''
    Given a list of token lists, count word frequencies and keep the
    top (max_vocab - len(SPECIAL_TOKENS)) most common words.

    Returns:
        word_to_idx  dict[str -> int]
        idx_to_word  dict[int -> str]
    '''
    counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)

    # reserve slots for special tokens, then fill with most common words
    vocab_words = SPECIAL_TOKENS + [
        word for word, _ in counter.most_common(max_vocab - len(SPECIAL_TOKENS))
    ]

    word_to_idx = {word: idx for idx, word in enumerate(vocab_words)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    return word_to_idx, idx_to_word


# ---------------------------------------------------------------------------
# Sequence encoding helpers
# ---------------------------------------------------------------------------
def encode_and_pad(tokens, word_to_idx, max_len):
    '''
    Convert a token list to a fixed-length list of integer indices.

    Steps:
      1. Map each token to its index (UNK_IDX if not in vocab).
      2. Truncate to max_len if too long.
      3. Pad with PAD_IDX on the right if too short.
    '''
    indices = [word_to_idx.get(t, UNK_IDX) for t in tokens]
    indices = indices[:max_len]                          # truncate
    indices += [PAD_IDX] * (max_len - len(indices))     # pad
    return indices


# ---------------------------------------------------------------------------
# Classification loader  (IMDB reviews)
# ---------------------------------------------------------------------------
def load_classification_data(data_dir, max_len=300, max_vocab=10000, random_seed=170):
    '''
    Load the IMDB sentiment dataset.

    Folder structure expected:
        data_dir/
            train/pos/*.txt
            train/neg/*.txt
            test/pos/*.txt
            test/neg/*.txt

    Parameters:
        data_dir   path to the text_classification folder
        max_len    maximum sequence length (longer reviews are truncated)
        max_vocab  maximum vocabulary size

    Returns:
        data         dict with train/test splits, X = index sequences, y = 0/1
        word_to_idx  vocabulary mapping word -> index
        idx_to_word  reverse mapping index -> word
    '''
    def read_split(split):
        '''Read all .txt files from one split (train or test).'''
        texts, labels = [], []
        for label_name, label_val in [('pos', 1), ('neg', 0)]:
            folder = os.path.join(data_dir, split, label_name)
            for fname in os.listdir(folder):
                if not fname.endswith('.txt'):
                    continue
                fpath = os.path.join(folder, fname)
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    raw = f.read()
                tokens = tokenize(clean_text(raw))
                texts.append(tokens)
                labels.append(label_val)
        return texts, labels

    print('Loading classification data...')
    train_texts, train_labels = read_split('train')
    test_texts,  test_labels  = read_split('test')

    print(f'  Train samples: {len(train_texts)}, Test samples: {len(test_texts)}')

    # build vocab from training data only (never peek at test)
    print('Building vocabulary...')
    word_to_idx, idx_to_word = build_vocab(train_texts, max_vocab=max_vocab)
    print(f'  Vocabulary size: {len(word_to_idx)}')

    train_X = [encode_and_pad(t, word_to_idx, max_len) for t in train_texts]
    test_X  = [encode_and_pad(t, word_to_idx, max_len) for t in test_texts]

    # shuffle training data so the model doesn't see all pos then all neg
    random.seed(random_seed)
    combined_train = list(zip(train_X, train_labels))
    random.shuffle(combined_train)
    train_X, train_labels = zip(*combined_train)
    train_X = list(train_X)
    train_labels = list(train_labels)

    data = {
        'train': {'X': train_X, 'y': train_labels},
        'test':  {'X': test_X,  'y': test_labels},
    }

    return data, word_to_idx, idx_to_word


# ---------------------------------------------------------------------------
# Generation loader  (short jokes)
# ---------------------------------------------------------------------------
def load_generation_data(data_dir, max_len=50, max_vocab=5000, random_seed=170):
    '''
    Load the short jokes dataset for next-word prediction.

    The jokes file is expected at:
        data_dir/shortjokes.csv   OR   data_dir/jokes.txt
    (one joke per line; CSV files use the last column as joke text)

    For generation, we frame the task as next-word prediction:
        X[i] = sequence[0 : n-1]   (all words except the last)
        y[i] = sequence[1 : n]     (all words except the first)

    Both X and y are padded to max_len.

    Parameters:
        data_dir   path to the text_generation folder
        max_len    maximum sequence length
        max_vocab  maximum vocabulary size

    Returns:
        data         dict — 90/10 train/test split after shuffling (seed random_seed)
        word_to_idx  vocabulary mapping (built from train split only)
        idx_to_word  reverse mapping
    '''

    # --- find the jokes file ---
    # accepts .csv, .txt, or an extensionless file named 'data'
    jokes_file = None
    for fname in os.listdir(data_dir):
        if fname.endswith('.csv') or fname.endswith('.txt') or fname == 'data':
            jokes_file = os.path.join(data_dir, fname)
            break

    if jokes_file is None:
        raise FileNotFoundError(f'No jokes file found in {data_dir}')

    print(f'Loading generation data from {os.path.basename(jokes_file)}...')

    # --- read jokes using csv module to handle quoted fields correctly ---
    import csv
    raw_jokes = []
    with open(jokes_file, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                continue  # skip header row ("ID", "Joke")
            if not row:
                continue  # skip blank lines
            # joke text is always the last column
            joke = row[-1].strip()
            if joke:
                raw_jokes.append(joke)

    print(f'  Total jokes: {len(raw_jokes)}')

    # --- clean and tokenize ---
    # add SOS and EOS around each joke so the model learns boundaries
    token_lists = []
    for joke in raw_jokes:
        tokens = [SOS_TOKEN] + tokenize(clean_text(joke)) + [EOS_TOKEN]
        if len(tokens) >= 2:
            token_lists.append(tokens)

    # --- shuffle and split (90/10) before vocab / encoding ---
    random.seed(random_seed)
    random.shuffle(token_lists)
    split_idx = int(len(token_lists) * 0.9)
    train_tokens = token_lists[:split_idx]
    test_tokens  = token_lists[split_idx:]

    # --- build vocab from training data only (never peek at test) ---
    print('Building vocabulary...')
    word_to_idx, idx_to_word = build_vocab(train_tokens, max_vocab=max_vocab)
    print(f'  Vocabulary size: {len(word_to_idx)}')

    # --- encode X/y pairs ---
    # X: tokens[0:-1]  (input to the model)
    # y: tokens[1:]    (what the model should predict at each step)
    def encode_split(tokens_list):
        X, y = [], []
        for tokens in tokens_list:
            X.append(encode_and_pad(tokens[:-1], word_to_idx, max_len))
            y.append(encode_and_pad(tokens[1:],  word_to_idx, max_len))
        return X, y

    train_X, train_y = encode_split(train_tokens)
    test_X,  test_y  = encode_split(test_tokens)

    print(f'  Train samples: {len(train_X)}, Test samples: {len(test_X)}')

    data = {
        'train': {'X': train_X, 'y': train_y},
        'test':  {'X': test_X,  'y': test_y},
    }

    return data, word_to_idx, idx_to_word


# ---------------------------------------------------------------------------
# Quick sanity check — run this file directly to verify everything loads
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys

    CLASSIFICATION_DIR = './ECS170_Spring_2026_Source_Code_Template/local_code/stage_4_code/stage_4_data/text_classification'
    GENERATION_DIR     = './ECS170_Spring_2026_Source_Code_Template/local_code/stage_4_code/stage_4_data/text_generation'

    print('=== Classification ===')
    clf_data, w2i, i2w = load_classification_data(CLASSIFICATION_DIR, max_len=300)
    sample_x = clf_data['train']['X'][0]
    sample_y = clf_data['train']['y'][0]
    print(f'  First sequence (first 10 indices): {sample_x[:10]}')
    print(f'  First label: {sample_y}')
    print(f'  Decoded: {[i2w[i] for i in sample_x[:10]]}')

    print()
    print('=== Generation ===')
    gen_data, w2i_g, i2w_g = load_generation_data(GENERATION_DIR, max_len=50)
    sample_x = gen_data['train']['X'][0]
    sample_y = gen_data['train']['y'][0]
    print(f'  Input  (first 10): {[i2w_g[i] for i in sample_x[:10]]}')
    print(f'  Target (first 10): {[i2w_g[i] for i in sample_y[:10]]}')

    print()
    print('All good! Dataset_Loader is ready.')