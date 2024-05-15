import torch


class MLP(torch.nn.Module):
  def __init__(self, input_dims, output_dims):
    super().__init__()

    self._model = torch.nn.Sequential(
      torch.nn.Linear(input_dims, 32),
      torch.nn.ReLU(),
      # torch.nn.Linear(32, 32),
      # torch.nn.ReLU(),
      torch.nn.Linear(32, output_dims)
    )

  def forward(self, x):
    return self._model(x)
