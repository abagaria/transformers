import pdb
import torch
import torch.nn.functional as F

from transformers.data.utils import CharDataLoader
from transformers.model.decoder import TransformerDecoder

batch_size = 32
seq_len = 20
embedding_size = 128
n_blocks = 1
n_heads = 4
dropout_p = 0.1
lr = 3e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_batches = 1000

print(f'Using device: {device}')

loader = CharDataLoader('shakespere.txt', batch_size, seq_len)

model = TransformerDecoder(
    n_blocks,
    embedding_size,
    n_heads,
    loader.vocab_size,
    seq_len,
    device,
    dropout_p)

model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for i in range(n_batches):
    x, y = loader.get_batch()
    x = x.to(device)
    y = y.to(device)

    logits = model(x)
    loss = F.cross_entropy(
        logits.reshape(-1, loader.vocab_size),
        y.reshape(-1))
    
    model.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(loss.item())
    
