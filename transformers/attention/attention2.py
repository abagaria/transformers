"""Efficient implementation of multi-headed attention."""

import pdb
import torch
import torch.nn as nn


class DecoderMultiHeadedAttention(nn.Module):
    def __init__(self, embedding_size, output_size, n_heads, dropout_p: float = 0.1):
        super().__init__()
        self._embedding_size = embedding_size
        self._output_size = output_size
        self._n_heads = n_heads

        self._qkv = nn.Linear(embedding_size, embedding_size * n_heads * 3, bias=False)
        self._fc = nn.Linear(embedding_size * n_heads, output_size)
        self._attn_dropout = nn.Dropout(dropout_p)
        self._fc_dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self._qkv(x)  # (B, T, C * N * 3)
        
        qkv2 = qkv.reshape(B, T, -1, 3)
        q, k, v = qkv2.chunk(3, dim=-1)

        # Reshape to (B, N, T, C)
        q = q.reshape(B, T, self._n_heads, C).transpose(1, 2)
        k = k.reshape(B, T, self._n_heads, C).transpose(1, 2)
        v = v.reshape(B, T, self._n_heads, C).transpose(1, 2)

        attention = q @ k.transpose(2, 3)  # (B, N, T, T)

        mask = torch.tril(torch.ones((T, T)))
        attention = torch.masked_fill(attention, mask == 0, float('-inf'))
        attention /= (C ** 0.5)

        weights = torch.softmax(attention, dim=-1)  # (B, N, T, T)
        weights = self._attn_dropout(weights)

        values = weights @ v  # (B, N, T, C)

        val_cat = values.transpose(1, 2).reshape(B, T, -1)
        assert val_cat.shape == (B, T, C * self._n_heads), val_cat.shape

        return self._fc_dropout(self._fc(val_cat))  # (B, T, D)


if __name__ == '__main__':
    batch_sz = 6
    seq_len = 4
    embed_sz = 8
    output_sz = 10
    X = torch.rand(batch_sz, seq_len, embed_sz)
    f = DecoderMultiHeadedAttention(embedding_size=embed_sz, output_size=output_sz, n_heads=9)
    y = f(X)
    print(f'Input shape: {X.shape} Output shape: {y.shape}')
