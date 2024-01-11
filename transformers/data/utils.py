"""Data loading and tokenization."""

import torch
import random


def load(path):
    with open(path, 'r') as f:
        text = f.read()
    return text


def create_char_tokenizer(text):
    chars = sorted(list(set(text)))
    char2int = {c: i for i, c in enumerate(chars)}
    int2char = {i: c for c, i in char2int.items()}
    encoder = lambda s: [char2int[c] for c in s]
    decoder = lambda l: ''.join([int2char[i] for i in l])
    return encoder, decoder, len(chars)


class CharDataLoader:
    def __init__(self, path, batch_size, seq_len) -> None:
        text = load(path)
        self.encoder, self.decoder, self.vocab_size = create_char_tokenizer(text)
        n_train = int(0.8 * len(text))
        
        self._train_split = text[:n_train]
        self._val_split = text[n_train:]

        self._batch_size = batch_size
        self._seq_len = seq_len

    def get_batch(self, split: str = 'train'):
        data = self._train_split if split == 'train' else self._val_split
        starting_idx = [random.randint(0, len(data) - self._seq_len - 1) for _ in range(self._batch_size)]
        text_blocks = [data[x: x + self._seq_len] for x in starting_idx]
        label_text_blocks = [data[x + 1: x + self._seq_len + 1] for x in starting_idx]
        token_blocks = [self.encoder(seq) for seq in text_blocks]
        label_text_blocks = [self.encoder(seq) for seq in label_text_blocks]
        return torch.tensor(token_blocks, dtype=torch.long),\
            torch.tensor(label_text_blocks, dtype=torch.long)


if __name__ == '__main__':
    loader = CharDataLoader('shakespere.txt', 4, 8)
    input, label = loader.get_batch()
    
    