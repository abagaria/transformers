import torch
import torch.nn as nn

from ..attention.attention2 import DecoderMultiHeadedAttention


class TransformerDecoderBlock(nn.Module):
    def __init__(self, embedding_size, n_heads, dropout_p: float = 0.1):
        super().__init__()

        self._mha = DecoderMultiHeadedAttention(
            embedding_size, embedding_size, n_heads, dropout_p)
        self._norm1 = nn.LayerNorm(embedding_size)
        self._classifier = nn.Sequential(
            nn.Linear(embedding_size, embedding_size * 4),
            nn.GELU(),
            nn.Linear(embedding_size * 4, embedding_size),
            nn.Dropout(dropout_p)
        )
        self._norm2 = nn.LayerNorm(embedding_size)

    def forward(self, x):
        x = x + self._mha(self._norm1(x))
        x = x + self._classifier(self._norm2(x))
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, n_blocks, embedding_size, n_heads,
                 vocab_size, seq_len, device, dropout_p: float = 0.1):
        super().__init__()
        self._embed = nn.Embedding(vocab_size, embedding_size)
        self._pos_embed = nn.Embedding(seq_len, embedding_size)
        self._model = nn.Sequential(
          *[TransformerDecoderBlock(embedding_size, n_heads, dropout_p) for _ in range(n_blocks)])
        self._norm = nn.LayerNorm(embedding_size)
        self._classifier = nn.Linear(embedding_size, vocab_size)
        self.device = device

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=self.device)
        embedding = self._embed(x) + self._pos_embed(pos)
        return self._classifier(self._norm(self._model(embedding)))
