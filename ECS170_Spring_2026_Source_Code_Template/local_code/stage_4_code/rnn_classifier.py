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
        dropout=0.3
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
            nonlinearity="tanh"
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)

        output, hidden = self.rnn(embedded)

        lengths = (x != self.pad_idx).sum(dim=1)
        lengths = torch.clamp(lengths, min=1)

        last_indices = lengths - 1

        batch_indices = torch.arange(x.size(0), device=x.device)

        last_output = output[batch_indices, last_indices]

        last_output = self.dropout(last_output)

        logits = self.fc(last_output)

        return logits