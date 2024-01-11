"""Multi-headed attention."""


import pdb
import torch
import torch.nn as nn


class DecoderMultiHeadedAttention(nn.Module):
    def __init__(self, embedding_size, output_dim, n_heads):
        super().__init__()

        self._embedding_size = embedding_size
        self._output_dim = output_dim
        self._n_heads = n_heads

        self._key = nn.Linear(embedding_size, embedding_size * n_heads, bias=False)
        self._query = nn.Linear(embedding_size, embedding_size * n_heads, bias=False)
        self._value = nn.Linear(embedding_size, embedding_size * n_heads, bias=False)

        self._fc = nn.Linear(embedding_size * n_heads, output_dim)

    def forward(self, x):
        B, T, C = x.shape

        key = self._key(x)      # (B, T, C * N)
        query = self._query(x)  # (B, T, C * N)
        value = self._value(x)  # (B, T, C * N)

        # Reshape to (B, T, C, N)
        key = key.reshape(B, T, C, self._n_heads)
        query = query.reshape(B, T, C, self._n_heads)
        value = value.reshape(B, T, C, self._n_heads)

        # Transpose from (B, T, C, N) to (B, N, T, C)
        key = key.permute(0, 3, 1, 2)
        query = query.permute(0, 3, 1, 2)
        value = value.permute(0, 3, 1, 2)

        # Apply the causal mask
        mask = torch.tril(torch.ones(T, T))
        agreement_matrix = query @ key.transpose(2, 3)  # (B, N, T, T)
        agreement_matrix = torch.masked_fill(agreement_matrix, mask == 0, float('-inf'))

        # Scale the attention matrix so that it has unit variance
        agreement_matrix /= (C ** 0.5)

        # Normalize the attention matrix to get a weight matrix
        weights = torch.softmax(agreement_matrix, dim=-1)  # (B, N, T, T)

        # Apply the weighted averaging to the values
        weighted_values = weights @ value  # (B, N, T, C)

        # Prepare the right shape for multi-headed attention
        output = weighted_values.transpose(1, 2).reshape((B, T, -1))  # (B, T, N * C)

        return self._fc(output)


if __name__ == '__main__':
    batch_sz = 6
    seq_len = 4
    embed_sz = 8
    output_sz = 10
    X = torch.rand(batch_sz, seq_len, embed_sz)
    f = DecoderMultiHeadedAttention(embedding_size=embed_sz, output_dim=output_sz, n_heads=10)
    y = f(X)
    print(f'Input shape: {X.shape} Output shape: {y.shape}')