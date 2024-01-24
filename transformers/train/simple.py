import pdb
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm

from transformers.data.utils import CharDataLoader
from transformers.model.decoder import TransformerDecoder


batch_size = 64
seq_len = 256
embedding_size = 384
n_blocks = 6
n_heads = 6
dropout_p = 0.4
lr = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_batches = 5_000
plotting_interval = 10

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

@torch.no_grad()
def compute_validation_loss(model, loader):
    val_x, val_y = loader.get_batch(split='val')
    val_x = val_x.to(device)
    val_y = val_y.to(device)
    logits = model(val_x)
    loss = F.cross_entropy(
        logits.reshape(-1, loader.vocab_size),
        val_y.reshape(-1))   
    return loss.item()

training_loss = []
validation_loss = []


for i in tqdm(range(n_batches), desc='Training'):
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

    if i % plotting_interval == 0:
        val_loss = compute_validation_loss(model, loader)
        training_loss.append(loss.item())
        validation_loss.append(val_loss)
    
        plt.plot(training_loss, label='Training Loss')
        plt.plot(validation_loss, label='Validation Loss')
        plt.grid()
        plt.legend()
        plt.savefig('loss.png')
        plt.close()
