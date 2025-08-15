# source: https://github.com/sfujim/TD3_BC
# https://arxiv.org/pdf/2106.06860.pdf

import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import minari
import gymnasium as gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import trange


TensorBatch = List[torch.Tensor]


@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    env: str = 'mujoco/hopper/expert-v0'  # OpenAI gym environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    n_episodes: int = 10  # How many episodes run during evaluation
    max_timesteps: int = int(1e6)  # Max time steps to run environment
    checkpoints_path: Optional[str] = None  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    # TD3
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount ffor
    expl_noise: float = 0.1  # Std of Gaussian exploration noise
    policy_noise: float = 0.2  # Noise added to target actor
    noise_clip: float = 0.5  # Range to clip target actor noise
    policy_freq: int = 2  # Frequency of delayed policy updates
    # TD3 + BC
    alpha: float = 2.5  # BC constraint
    normalize: bool = False
    normalize_reward: bool = False

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
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
    def __init__(
        self, state_dim: int, action_dim: int, buffer_size: int, normalize_reward: bool = False
    ):
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

    def normalize_rewards(self, eps: float = 1e-3):
        mean = self.rewards.mean()
        std = self.rewards.std() + eps
        self.rewards = (self.rewards - mean) / std

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = torch.from_numpy(self.states[indices])
        actions = torch.from_numpy(self.actions[indices])
        rewards = torch.from_numpy(self.rewards[indices])
        next_states = torch.from_numpy(self.next_states[indices])
        dones = torch.from_numpy(self.dones[indices])
        return [states, actions, rewards, next_states, dones]


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Critic, self).__init__()

        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


class TD3_BC(object):
    def __init__(
        self,
        actor: Actor,
        critic: Critic,
        discount: float,
        tau: float,
        policy_noise: float,
        noise_clip: float,
        policy_freq: int,
        alpha: float,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        device: str,
    ):
        self.actor = actor
        self.critic = critic
        self.actor_target = copy.deepcopy(actor)
        self.critic_target = copy.deepcopy(critic)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha

        self.device = device
        self.total_it = 0

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        state, action, reward, next_state, done = batch
        # Update critic
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            noise = (
                torch.randn_like(action)
                * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            next_action = (next_action + noise).clamp(-1, 1)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + done * self.discount * target_Q

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        if self.total_it % self.policy_freq == 0:
            pi = self.actor(state)
            Q = self.critic.l3(self.critic.l2(F.relu(self.critic.l1(torch.cat([state, pi], 1)))))
            lmbda = self.alpha / Q.abs().mean().detach()
            actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, action)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
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
            state_normalized_tensor = torch.from_numpy(state_normalized).float().to(device)
            action = actor(state_normalized_tensor).cpu().numpy()
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
    eval_env = gym.make(config.env)

    dataset = minari.load_dataset(config.env, download=True, force_download=False)

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

    # Initialize TD3_BC
    actor = Actor(state_dim, action_dim, config.max_action).to(config.device)
    critic = Critic(state_dim, action_dim).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)

    kwargs = {
        "actor": actor,
        "critic": critic,
        "actor_optimizer": actor_optimizer,
        "critic_optimizer": critic_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "policy_noise": config.policy_noise,
        "noise_clip": config.noise_clip,
        "policy_freq": config.policy_freq,
        "alpha": config.alpha,
        "device": config.device,
    }

    print("---------------------------------------")
    print(f"Training TD3 + BC")
    print("---------------------------------------")

    # Initialize actor
    trainer = TD3_BC(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    evaluations = []
    for t in trange(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        if config.normalize:
            states, actions, rewards, next_states, dones = batch
            states_normalized = normalize_states(states.cpu().numpy(), state_mean, state_std)
            next_states_normalized = normalize_states(next_states.cpu().numpy(), state_mean, state_std)
            states = torch.from_numpy(states_normalized).to(config.device)
            next_states = torch.from_numpy(next_states_normalized).to(config.device)
            batch = [states, actions, rewards, next_states, dones]

        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)

        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            eval_scores = eval_actor(
                eval_env,
                actor,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
                state_mean=state_mean,
                state_std=state_std,
            )
            eval_score = eval_scores.mean()
            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"{eval_score:.3f} , Minari score mean: {eval_score:.3f}"
            )
            print("---------------------------------------")

            if config.checkpoints_path is not None:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )

if __name__ == "__main__":
    train()