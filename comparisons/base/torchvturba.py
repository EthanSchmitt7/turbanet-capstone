from time import time

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import torch
import torch.nn as nn
from flax import linen
from jax.lib import xla_bridge
from turbanet import TurbaTrainState


def cross_entropy(logprobs, labels):
    one_hot_labels = jax.nn.one_hot(labels, logprobs.shape[1])
    return -jnp.mean(jnp.sum(one_hot_labels * logprobs, axis=-1))


def cross_entropy_turba(params, batch, apply_fn):
    log_probs = apply_fn({"params": params}, batch["input"])
    labels = jax.nn.one_hot(batch["output"], log_probs.shape[1])
    loss = -jnp.mean(jnp.sum(labels * log_probs, axis=-1))
    return loss, labels


class JaxModel(linen.Module):
    hidden_layers: int = 1
    hidden_dim: int = 32

    @linen.compact
    def __call__(self, x):
        for layer in range(self.hidden_layers):
            x = linen.Dense(self.hidden_dim)(x)
            x = linen.relu(x)
        x = linen.Dense(2)(x)
        x = linen.log_softmax(x)
        return x


classifier_fns = JaxModel()


class TorchModel(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int):
        super(TorchModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.stack = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            *(nn.Linear(hidden_size, hidden_size), nn.ReLU()) * (num_layers - 1),
            nn.Linear(hidden_size, 2),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.stack(x)


# GENERAL INPUTS
GPU = False
seed = 0

# NETWORK SHAPE INPUTS
hidden_size = 8
num_layers = 3

# TRAINING INPUTS
swarm_size = 100
epochs = 1000
lr = 0.0001
dataset_size = 10000

# Set numpy/torch/flax seeds
np.random.seed(seed)
torch.manual_seed(seed)


# Create torch model
torch_models = [TorchModel(hidden_size, num_layers) for _ in range(swarm_size)]
torch_optimizers = [
    torch.optim.Adam(torch_model.parameters(), lr=lr) for torch_model in torch_models
]

# Create Turba model
turba_model = JaxModel(hidden_size, num_layers)
turba_state = TurbaTrainState.swarm(turba_model, swarm_size, 2, learning_rate=lr)

# Set torch to use GPU if available
device = torch.device("cpu")
if GPU and torch.cuda.is_available():
    device = torch.device("cuda")
    for torch_model in torch_models:
        torch_model.to(device)

# Set Turba to use GPU if available
if GPU and xla_bridge.get_backend().platform != "gpu":
    raise RuntimeError("GPU support not available for Turba.")


# Create random data
def make_spirals(n_samples, noise_std=0.0, rotations=1.0):
    ts = jnp.linspace(0, 1, n_samples)
    rs = ts**0.5
    thetas = rs * rotations * 2 * np.pi
    signs = np.random.randint(0, 2, (n_samples,)) * 2 - 1
    labels = (signs > 0).astype(int)

    xs = rs * signs * jnp.cos(thetas) + np.random.randn(n_samples) * noise_std
    ys = rs * signs * jnp.sin(thetas) + np.random.randn(n_samples) * noise_std
    points = jnp.stack([xs, ys], axis=1)
    return points, labels


points, labels = make_spirals(dataset_size, noise_std=0.05)


# Train torch model
# Convert to torch tensors
X_train_torch = torch.from_numpy(np.array(points)).float()
y_train_torch = torch.from_numpy(np.array(labels))

# Move to GPU if available
if GPU:
    X_train_torch = X_train_torch.to(device)
    y_train_torch = y_train_torch.to(device)

start = time()
for torch_model, torch_optimizer in zip(torch_models, torch_optimizers):
    torch_model.train()
    for epoch in range(epochs):
        torch_optimizer.zero_grad()
        y_pred = torch_model(X_train_torch)
        loss = torch.nn.functional.cross_entropy(y_pred, y_train_torch)
        loss.backward()
        torch_optimizer.step()

print(f"torch time: {time() - start}")

# Train Turba model
# Convert to jnp arrays
X_train_turba = jnp.array(
    np.expand_dims(points, axis=0).repeat(swarm_size, axis=0), dtype=jnp.float32
)
y_train_turba = jnp.array(np.expand_dims(labels, axis=0).repeat(swarm_size, axis=0))

start = time()
for epoch in range(epochs):
    turba_state, _, _ = turba_state.train(X_train_turba, y_train_turba, cross_entropy_turba)

print(f"turba time: {time() - start}")


# Train jax model
def loss_fn(params, batch):
    logits = classifier_fns.apply({"params": params}, batch[0])
    loss = jnp.mean(cross_entropy(logits, batch[1]))
    return loss, logits


loss_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

optimizer = optax.adam(learning_rate=lr)


def init_fn(input_shape, seed):
    rng = jr.PRNGKey(seed)
    dummy_input = jnp.ones((1, input_shape))
    params = classifier_fns.init(rng, dummy_input)["params"]
    opt_state = optimizer.init(params)
    return params, opt_state


@jax.jit
def train_step_fn(params, opt_state, batch):
    (loss, grad), output = loss_and_grad_fn(params, batch)
    updates, opt_state = optimizer.update(grad, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, output


parallel_init_fn = jax.vmap(init_fn, in_axes=(None, 0))
parallel_train_step_fn = jax.vmap(train_step_fn, in_axes=(0, 0, None))

seeds = jnp.linspace(0, swarm_size - 1, swarm_size).astype(int)
model_states, opt_states = parallel_init_fn(2, seeds)
start = time()
for i in range(epochs):
    model_states, opt_states, _ = parallel_train_step_fn(
        model_states, opt_states, (points, labels)
    )

print(f"jax time: {time() - start}")
