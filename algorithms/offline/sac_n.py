# Inspired by:
# 1. paper for SAC-N: https://arxiv.org/abs/2110.01548
# 2. implementation: https://github.com/snu-mllab/EDAC

import math
import os
import random
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import gymnasium as gym
import minari
import pickle
import numpy as np
import pyrallis
import torch
import torch.nn as nn
from torch.distributions import Normal
from tqdm import trange
import socket
import json


HOME = os.path.dirname(os.path.realpath(__file__))


@dataclass
class TrainConfig:
    # wandb params
    project: str = "CORL"
    group: str = "SAC-N"
    name: str = "SAC-N"
    # model params
    hidden_dim: int = 256
    num_critics: int = 10
    gamma: float = 0.99
    tau: float = 5e-3
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    max_action: float = 1.0
    # training params
    buffer_size: int = 1_000_000
    env_name: str = "mujoco/hopper/expert-v0"
    batch_size: int = 256
    num_epochs: int = 3000
    num_updates_on_epoch: int = 1000
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 5
    normalize: bool = False
    normalize_reward: bool = False
    # general params
    seed: int = 42
    eval_seed: int = 69
    checkpoints_path: Optional[str] = None
    device: str = "cuda"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def set_seed(seed: int, env: Optional[gym.Env] = None):
    if env is not None:
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def compute_mean_std(states: np.ndarray, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std

def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, buffer_size: int, normalize_reward: bool = False):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0
        self._normalize_reward = normalize_reward

        self.states = np.zeros(
            (buffer_size, state_dim), dtype=np.float32
        )
        self.actions = np.zeros(
            (buffer_size, action_dim), dtype=np.float32
        )
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.next_states = np.zeros(
            (buffer_size, state_dim), dtype=np.float32
        )
        self.dones = np.zeros((buffer_size, 1), dtype=np.float32)

    # Loads data in minari, i.e. from Dict[str, np.array].
    def load_minari_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self.states[:n_transitions] = data["observations"]
        self.actions[:n_transitions] = data["actions"]
        self.rewards[:n_transitions] = data["rewards"][..., None]
        self.next_states[:n_transitions] = data["next_observations"]
        self.dones[:n_transitions] = data["terminals"][..., None]
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)
        if self._normalize_reward:
            self.normalize_rewards()

        print(f"Dataset size: {n_transitions}")
    
    def normalize_rewards(self, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes mean and std of rewards and normalizes them.
        """
        reward_mean = self.rewards.mean(axis=0)
        reward_std = self.rewards.std(axis=0) + eps
        self.rewards = (self.rewards - reward_mean) / reward_std
        return reward_mean, reward_std

    def sample(self, batch_size: int) -> Tuple[torch.Tensor]:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        return (
            torch.from_numpy(self.states[indices]),
            torch.from_numpy(self.actions[indices]),
            torch.from_numpy(self.rewards[indices]),
            torch.from_numpy(self.next_states[indices]),
            torch.from_numpy(self.dones[indices]),
        )


class MLP(nn.Module):
    def __init__(
        self,
        dims: Sequence[int],
        activation_fn: Callable = nn.ReLU,
        output_activation_fn: Callable = None,
        layernorm: bool = False,
    ):
        super().__init__()
        self._dims = dims
        self._activation_fn = activation_fn
        self._output_activation_fn = output_activation_fn
        self._layernorm = layernorm

        self._layers = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            self._layers.append(nn.Linear(in_dim, out_dim))
            if layernorm:
                self._layers.append(nn.LayerNorm(out_dim))
            if i < len(dims) - 2:
                self._layers.append(activation_fn())
            elif output_activation_fn is not None:
                self._layers.append(output_activation_fn())

        self.net = nn.Sequential(*self._layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TanhNormal(Normal):
    def __init__(self, loc, scale):
        super().__init__(loc, scale)
        self.transforms = TanhTransform()

    @property
    def mean(self):
        return self.transforms.inv(super().mean)

    def rsample(self, sample_shape=torch.Size()):
        x = super().rsample(sample_shape)
        return self.transforms(x)


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float = 1.0):
        super().__init__()
        self.log_std_min = -20
        self.log_std_max = 2
        self.max_action = max_action
        self.trunk = MLP([state_dim, 256, 256, 256])
        self.mu = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
        with_log_prob: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(state)
        mu = self.mu(hidden)
        log_std = self.log_std(hidden).clamp(self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        if deterministic:
            action = torch.tanh(mu)
        else:
            action = dist.rsample()

        if with_log_prob:
            log_prob = dist.log_prob(action).sum(axis=-1)
            log_prob -= (2 * (math.log(2) - action.tanh()).sum(axis=-1))
        else:
            log_prob = None

        action = self.max_action * action
        return action, log_prob


class EnsembleCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, num_critics: int):
        super().__init__()
        self.num_critics = num_critics
        self.critics = nn.ModuleList()
        for i in range(num_critics):
            self.critics.append(MLP([state_dim + action_dim, 256, 256, 256, 1]))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state_action = torch.cat([state, action], dim=-1)
        q_values = [critic(state_action) for critic in self.critics]
        q_values = torch.cat(q_values, dim=-1)
        return q_values


class SACN:
    def __init__(
        self,
        actor: Actor,
        actor_optimizer: torch.optim.Optimizer,
        critic: EnsembleCritic,
        critic_optimizer: torch.optim.Optimizer,
        alpha_optimizer: torch.optim.Optimizer,
        alpha: Union[float, torch.Tensor],
        tau: float,
        gamma: float,
        num_critics: int,
        device: str = "cpu",
    ):
        self.actor = actor
        self.critic = critic
        self.target_critic = deepcopy(self.critic)
        self.target_critic.requires_grad_(False)
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.alpha_optimizer = alpha_optimizer
        self.alpha = alpha
        self.tau = tau
        self.gamma = gamma
        self.num_critics = num_critics
        self.device = device
        self.total_updates = 0

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.total_updates += 1
        states, actions, rewards, next_states, dones = (
            batch["states"],
            batch["actions"],
            batch["rewards"],
            batch["next_states"],
            batch["dones"],
        )

        # Update alpha
        with torch.no_grad():
            _, log_pi = self.actor(states)
            alpha_loss = -self.alpha * (log_pi + self.actor.action_dim).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Update critic
        with torch.no_grad():
            next_actions, next_log_pi = self.actor(next_states)
            q_values_next = self.target_critic(next_states, next_actions)
            q_values_next = (
                torch.mean(q_values_next, dim=-1, keepdim=True)
                - self.alpha * next_log_pi
            )
            q_target = rewards + (1.0 - dones) * self.gamma * q_values_next
        q_values = self.critic(states, actions)
        q_loss = torch.mean((q_values - q_target) ** 2)
        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        _, log_pi = self.actor(states)
        q_values_actor = self.critic(states, log_pi)
        actor_loss = (self.alpha * log_pi - torch.mean(q_values_actor, dim=-1)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target critic
        with torch.no_grad():
            for param, target_param in zip(
                self.critic.parameters(), self.target_critic.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

        return {
            "q_loss": q_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
        }


@torch.no_grad()
def eval_actor(
    env: gym.Env,
    actor: Actor,
    device: str,
    n_episodes: int,
    seed: int,
    state_mean: Union[np.ndarray, float],
    state_std: Union[np.ndarray, float],
) -> np.ndarray:
    actor.eval()
    eval_returns = []
    set_seed(seed, env)
    for i in range(n_episodes):
        state, info = env.reset(seed=seed + 100)
        state_normalized = (state - state_mean) / state_std
        done = False
        total_reward = 0.0
        while not done:
            state_normalized = torch.from_numpy(state_normalized).float().to(device)
            action, _ = actor(state_normalized, deterministic=True)
            action = action.cpu().numpy()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state
            state_normalized = (state - state_mean) / state_std

        eval_returns.append(total_reward)

    actor.train()
    return np.asarray(eval_returns)


@pyrallis.wrap()
def train(config: TrainConfig):
    set_seed(config.seed)
    eval_env = gym.make(config.env_name)

    dataset = minari.load_dataset(config.env_name, download=True, force_download=False)

    state_dim = dataset.observations.shape[1]
    action_dim = dataset.actions.shape[1]

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset.observations, eps=1e-3)
    else:
        state_mean, state_std = 0.0, 1.0

    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        normalize_reward=config.normalize_reward,
    )
    replay_buffer.load_minari_dataset(dataset)

    # Initializing actors and critics
    actor = Actor(state_dim, action_dim, config.max_action).to(config.device)
    actor_optimizer = torch.optim.Adam(
        actor.parameters(), lr=config.actor_learning_rate
    )

    critic = EnsembleCritic(state_dim, action_dim, config.num_critics).to(config.device)
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=config.critic_learning_rate
    )

    alpha = torch.tensor(1.0, requires_grad=True, device=config.device)
    alpha_optimizer = torch.optim.Adam(
        [alpha], lr=config.alpha_learning_rate
    )

    print("---------------------------------------")
    print(f"Training SAC-N, Env: {config.env_name}")
    print("---------------------------------------")

    kwargs = {
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "critic": critic,
        "critic_optimizer": critic_optimizer,
        "alpha_optimizer": alpha_optimizer,
        "alpha": alpha,
        "tau": config.tau,
        "gamma": config.gamma,
        "num_critics": config.num_critics,
        "device": config.device,
    }
    # Initialize policy
    trainer = SACN(**kwargs)

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    total_updates = 0.0
    for epoch in trange(config.num_epochs, desc="Training"):
        for _ in trange(config.num_updates_on_epoch, desc="Epoch", leave=False):
            batch = replay_buffer.sample(config.batch_size)
            batch = {k: v.to(config.device) for k, v in batch.items()}
            update_info = trainer.update(batch)
            total_updates += 1

        if epoch % config.eval_every == 0 or epoch == config.num_epochs - 1:
            eval_returns = eval_actor(
                env=eval_env,
                actor=actor,
                n_episodes=config.eval_episodes,
                seed=config.eval_seed,
                device=config.device,
                state_mean=state_mean,
                state_std=state_std,
            )

            eval_score = eval_returns.mean()
            print("---------------------------------------")
            print(
                f"Evaluation over {config.eval_episodes} episodes: "
                f"{eval_score:.3f} , Minari score mean: {eval_score:.3f}"
            )
            print("---------------------------------------")

            if config.checkpoints_path is not None:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"{epoch}.pt"),
                )


if __name__ == "__main__":
    train()