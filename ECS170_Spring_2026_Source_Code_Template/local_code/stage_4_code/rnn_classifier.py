import torch
import torch.nn as nn


class RNNClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        output_dim,
        pad_idx,
        dropout=0.5
    ):
        super().__init__()

        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx
        )

        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
            nonlinearity="tanh"
        )

        self.dropout = nn.Dropout(dropout)

        # bidirectional output = hidden_dim * 2
        # mean pooling + max pooling = hidden_dim * 4
        self.fc = nn.Linear(hidden_dim * 4, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)

        outputs, hidden = self.rnn(embedded)
        # outputs: [batch, seq_len, hidden_dim * 2]

        mask = (x != self.pad_idx).unsqueeze(-1)
        masked_outputs = outputs * mask

        lengths = mask.sum(dim=1).clamp(min=1)

        mean_pool = masked_outputs.sum(dim=1) / lengths

        outputs_for_max = outputs.masked_fill(~mask, -1e9)
        max_pool, _ = outputs_for_max.max(dim=1)

        pooled = torch.cat([mean_pool, max_pool], dim=1)

        pooled = self.dropout(pooled)

        logits = self.fc(pooled)

        return logits