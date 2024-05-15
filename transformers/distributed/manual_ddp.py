"""Implement data-parallel training without using PyTorch's DistributedDataParallel."""

import os
import torch
import random
import matplotlib.pyplot as plt
import torch.distributed as dist
import torch.multiprocessing as mp

from model.mlp import MLP


lr = 1e-3
batch_size = 64
num_features = 4
num_batches = 1000


input_features = torch.randn(640, num_features)
input_labels = torch.rand(640)


def init_process_group(rank, world_size):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12355'
  dist.init_process_group('nccl', rank=rank, world_size=world_size)
  run(rank, world_size)


def create_model(rank) -> torch.nn.Module:
  model = MLP(num_features, 1).to(f'cuda:{rank}')
  replicate_model(model)
  return model
  

def replicate_model(model: torch.nn.Module):
  """Master rank (rank=0) will braodcast this model parameters to everyone."""
  for p in model.parameters():
    dist.broadcast(p.data, src=0)


def sample_data(rank, world_size):
  """Dummy version of the distributed data sampler."""
  micro_batch_size = batch_size // world_size
  idx = random.sample(range(len(input_features)), k=micro_batch_size)
  X = input_features[idx, :].to(f'cuda:{rank}')
  Y = input_labels[idx].to(f'cuda:{rank}')
  return X, Y


def sgd_step(model: torch.nn.Module, learning_rate: float):
  for p in model.parameters():
    p.data -= (learning_rate * p.grad.data)


def sync_gradients(model: torch.nn.Module, world_size: int):
  """Sync the gradients across all devices. This is naive and will lead to a
  big pipeline bubble. It would be better to interleave communication and 
  computation by spawning an all_reduce after each layer's backward pass."""
  for p in model.parameters():
    if p.grad is not None:
      dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
      p.grad.data /= world_size


def run_training_batch(model, rank, world_size):
  X, Y = sample_data(rank, world_size)
  logits = model(X)
  loss = torch.nn.functional.mse_loss(logits.squeeze(), Y)
  
  model.zero_grad()
  loss.backward()
  sync_gradients(model, world_size)
  sgd_step(model, learning_rate=lr)

  return loss.item()


def run_training_epoch(model, rank, world_size):
    losses = []
    for _ in range(10):
      losses.append(run_training_batch(model, rank, world_size))
    return sum(losses) / len(losses)


def run(rank, world_size):
  losses = []
  param_inconsistencies = []
  gradient_inconsistencies = []
  model = create_model(rank)
  for _ in range(num_batches):
    loss = run_training_epoch(model, rank, world_size)
    losses.append(loss)
    
    inconsistency = log_inconsistency(model, world_size)
    param_inconsistencies.append(inconsistency[0])
    gradient_inconsistencies.append(inconsistency[1])

  if rank == 0:
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('manual_ddp_loss_curve.png')
    plt.close()

    plt.plot(param_inconsistencies)
    plt.ylabel('Parameter Inconsistency')
    plt.savefig('param_inconsistency.png')
    plt.close()

    plt.plot(gradient_inconsistencies)
    plt.ylabel('Gradient Inconsistency')
    plt.savefig('gradient_inconsistency.png')
    plt.close()

def log_inconsistency(model: torch.tensor, world_size: int):
  """Log the norm of the difference b/w the parameter and grad values."""
  param_diffs = []
  gradient_diffs = []
  for p in model.parameters():
    if p is not None:
      param_list = [torch.empty_like(p.data) for _ in range(world_size)]
      dist.all_gather(param_list, p.data)
      param_diffs.append(
        torch.stack(param_list).std(0).mean().item())

      grad_list = [torch.empty_like(p.grad.data) for _ in range(world_size)]
      dist.all_gather(grad_list, p.grad.data)
      gradient_diffs.append(
        torch.stack(grad_list).std(0).mean().item())

  return sum(param_diffs) / len(param_diffs), \
    sum(gradient_diffs) / len(gradient_diffs)


if __name__ == '__main__':
  world_size = torch.cuda.device_count()
  print(f'Spawning job across {world_size} devices')
  mp.spawn(init_process_group, args=(world_size,), nprocs=world_size)
