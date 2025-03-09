from time import time

import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn
from flax import linen
from jax.lib import xla_bridge
from turbanet import TurbaTrainState, l2_loss


class TorchModel(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int, num_layers: int):
        super(TorchModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            *(nn.Linear(hidden_size, hidden_size), nn.ReLU()) * (num_layers - 1),
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.stack(x)


class TurbaModel(linen.Module):
    input_size: int
    output_size: int
    hidden_size: int
    num_layers: int

    @linen.compact
    def __call__(self, x):
        x = linen.Dense(features=self.hidden_size)(x)
        for _ in range(self.num_layers - 1):
            x = linen.relu(x)
            x = linen.Dense(features=self.hidden_size)(x)
        x = linen.relu(x)
        x = linen.Dense(features=self.output_size)(x)
        x = linen.relu(x)
        return x


# GENERAL INPUTS
GPU = False
seed = 0

# NETWORK SHAPE INPUTS
input_size = 6
output_size = 3
hidden_size = 8
num_layers = 3

# TRAINING INPUTS
swarm_size = 10
epochs = 10
batch_size = 64
batches = 100
dataset_size = batches * batch_size
lr = 0.01

# Set numpy/torch/flax seeds
np.random.seed(seed)
torch.manual_seed(seed)

# Create torch model
torch_model = TorchModel(input_size, output_size, hidden_size, num_layers)
torch_optimizer = torch.optim.Adam(torch_model.parameters(), lr=lr)

# Create Turba model
turba_model = TurbaModel(input_size, output_size, hidden_size, num_layers)
turba_state = TurbaTrainState.swarm(turba_model, swarm_size, input_size, learning_rate=lr)

# Set torch to use GPU if available
device = torch.device("cpu")
if GPU and torch.cuda.is_available():
    device = torch.device("cuda")
    torch_model.to(device)

# Set Turba to use GPU if available
if GPU and xla_bridge.get_backend().platform != "gpu":
    raise RuntimeError("GPU support not available for Turba.")

# Create random data
X_train = np.random.rand(batches, batch_size, input_size)
y_train = np.random.rand(batches, batch_size, output_size)

# Convert to torch tensors
X_train_torch = torch.from_numpy(X_train).float()
y_train_torch = torch.from_numpy(y_train).float()

# Move to GPU if available
if GPU:
    X_train_torch = X_train_torch.to(device)
    y_train_torch = y_train_torch.to(device)

# Convert to jnp arrays
X_train_turba = jnp.array(
    np.expand_dims(X_train, axis=1).repeat(swarm_size, axis=1), dtype=jnp.float32
)
y_train_turba = jnp.array(
    np.expand_dims(y_train, axis=1).repeat(swarm_size, axis=1), dtype=jnp.float32
)

# Train torch model
start = time()
torch_model.train()
for epoch in range(epochs):
    for i in range(batches):
        torch_optimizer.zero_grad()
        y_pred = torch_model(X_train_torch[i])
        loss = torch.nn.functional.mse_loss(y_pred, y_train_torch[i])
        loss.backward()
        torch_optimizer.step()

print(f"torch time: {time() - start}")

# Train Turba model
start = time()
for epoch in range(epochs):
    for i in range(batches):
        turba_state, _, _ = turba_state.train(X_train_turba[i], y_train_turba[i], l2_loss)

print(f"turba time: {time() - start}")
