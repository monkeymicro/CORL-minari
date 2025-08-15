# source: https://github.com/young-geng/CQL/tree/934b0e8354ca431d6c083c4e3a29df88d4b0a24d
# https://arxiv.org/pdf/2006.04779.pdf
import os
import random
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import argparse
from tqdm import trange
import pickle
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import socket
import json
from torch.distributions import Normal, TanhTransform, TransformedDistribution
import minari
import gymnasium as gym

TensorBatch = List[torch.Tensor]
HOME = os.path.dirname(os.path.realpath(__file__))

EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    # env: str = 'mujoco/inverteddoublependulum/expert-v0'  # OpenAI gym environment name
    env: str = 'mujoco/hopper/expert-v0'
    seed: int = 1  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    n_episodes: int = 10  # How many episodes run during evaluation
    max_timesteps: int = int(1e6)  # Max time steps to run environment
    checkpoints_path: Optional[str] = None  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load

    # CQL
    buffer_size: int = 1000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    tau: float = 0.005  # Target network update rate
    actor_learning_rate: float = 3e-4  # Actor learning rate
    critic_learning_rate: float = 3e-4  # Critic learning rate
    alpha_learning_rate: float = 3e-4  # Alpha learning rate
    alpha_init: float = 1.0
    alpha_autotune: bool = True
    target_entropy: Optional[float] = None

    cql_alpha: float = 5.0
    cql_n_actions: int = 10
    cql_log_alpha: float = -1.0
    cql_target_action_gap: float = 1.0
    cql_clip_diff_min: float = -200
    cql_clip_diff_max: float = 20
    # Additional
    normalize: bool = False  # Normalize states
    normalize_reward: bool = False
    port:int = 11024


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


def set_seed(
    seed: int,
):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_minari_dataset(self, data: Dict[str, np.ndarray], normalize_reward: bool):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        if normalize_reward:
            self.normalize_rewards()

        print(f"Dataset size: {n_transitions}")

    def normalize_rewards(self, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes mean and std of rewards and normalizes them.
        """
        reward_mean = self._rewards.mean(dim=0, keepdim=True)
        reward_std = self._rewards.std(dim=0, keepdim=True) + eps
        self._rewards = (self._rewards - reward_mean) / reward_std
        return reward_mean.cpu().numpy(), reward_std.cpu().numpy()

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError


class ReparameterizedTanhGaussian(nn.Module):
    def __init__(
        self,
        actor: nn.Module,
        log_std_min: float = LOG_STD_MIN,
        log_std_max: float = LOG_STD_MAX,
    ):
        super().__init__()
        self.actor = actor
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state: torch.Tensor):
        mean, log_std = self.actor(state)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        tanh_normal = Normal(mean, std)
        return TransformedDistribution(tanh_normal, TanhTransform())


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(state)
        mean = self.mean(hidden)
        log_std = self.log_std(hidden)
        return mean, log_std


class VectorizedCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_critics: int,
    ):
        super().__init__()
        self.critics = nn.ModuleList()
        for i in range(num_critics):
            self.critics.append(
                nn.Sequential(
                    nn.Linear(state_dim + action_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )
            )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state_action = torch.cat([state, action], dim=-1)
        q_values = [critic(state_action) for critic in self.critics]
        q_values = torch.stack(q_values, dim=0)
        return q_values


class ContinuousCQL:
    def __init__(
        self,
        actor: Actor,
        actor_optimizer: torch.optim.Optimizer,
        critic: VectorizedCritic,
        critic_optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha_learning_rate: float = 1e-4,
        alpha_init: float = 1.0,
        alpha_autotune: bool = True,
        target_entropy: Optional[float] = None,
        cql_alpha: float = 5.0,
        cql_n_actions: int = 10,
        cql_log_alpha: float = -1.0,
        cql_target_action_gap: float = 1.0,
        cql_clip_diff_min: float = -200,
        cql_clip_diff_max: float = 20,
        device: str = "cpu",
    ):
        self.device = device

        self.actor = actor
        self.actor_optimizer = actor_optimizer

        self.critic = critic
        self.critic_target = deepcopy(self.critic)
        self.critic_optimizer = critic_optimizer

        self.log_alpha = torch.nn.Parameter(torch.tensor(np.log(alpha_init), device=device, requires_grad=True))
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_learning_rate
        )

        self.cql_log_alpha = torch.nn.Parameter(torch.tensor([cql_log_alpha], device=device, requires_grad=True))
        self.cql_log_alpha_optimizer = torch.optim.Adam(
            [self.cql_log_alpha], lr=alpha_learning_rate
        )

        self.cql_alpha = cql_alpha
        self.cql_n_actions = cql_n_actions
        self.cql_target_action_gap = cql_target_action_gap
        self.cql_clip_diff_min = cql_clip_diff_min
        self.cql_clip_diff_max = cql_clip_diff_max

        self.tau = tau
        self.gamma = gamma
        self.alpha_autotune = alpha_autotune
        self.target_entropy = target_entropy
        if self.target_entropy is None:
            self.target_entropy = -torch.prod(
                torch.Tensor(self.actor.action_dim).to(self.device)
            ).item()

        self.total_it = 0

    def _alpha_loss(self, states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, action_log_prob = self.actor(states)
        loss = (-self.log_alpha * (action_log_prob + self.target_entropy)).mean()
        return loss

    def _critic_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        with torch.no_grad():
            next_actions, next_action_log_prob = self.actor(next_states)
            q_target_next_states = self.critic_target(next_states, next_actions)
            q_target_next_states = torch.min(q_target_next_states, dim=0)[0]
            q_target = rewards + (1.0 - dones) * self.gamma * (
                q_target_next_states - self.alpha.detach() * next_action_log_prob
            )

        q_pred = self.critic(states, actions)
        q_pred_min = torch.min(q_pred, dim=0)[0]
        critic_loss = F.mse_loss(q_pred, q_target.detach().expand(q_pred.shape))

        # CQL part
        cql_random_actions = torch.empty(
            (self.cql_n_actions, states.shape[0], actions.shape[1]),
            dtype=torch.float32,
            device=self.device,
        ).uniform_(-1, 1)
        cql_random_q = self.critic(
            states.unsqueeze(0).repeat(self.cql_n_actions, 1, 1).view(-1, states.shape[1]),
            cql_random_actions.view(-1, actions.shape[1]),
        ).view(self.critic.num_critics, self.cql_n_actions, states.shape[0], 1)
        cql_random_q = torch.min(cql_random_q, dim=0)[0]

        cql_current_actions, cql_current_action_log_prob = self.actor(states)
        cql_current_q = self.critic(states, cql_current_actions)
        cql_current_q = torch.min(cql_current_q, dim=0)[0]

        cql_next_actions, cql_next_action_log_prob = self.actor(next_states)
        cql_next_q = self.critic(next_states, cql_next_actions)
        cql_next_q = torch.min(cql_next_q, dim=0)[0]

        cql_diff_1 = (
            torch.logsumexp(cql_random_q - self.cql_log_alpha.exp(), dim=0)
        ) * self.cql_log_alpha.exp()
        cql_diff_2 = (
            torch.logsumexp(cql_next_q - self.cql_log_alpha.exp(), dim=0)
        ) * self.cql_log_alpha.exp()

        cql_min_q_loss = (cql_diff_1 + cql_diff_2) - q_pred_min.mean()
        cql_loss = (cql_min_q_loss + self.cql_alpha * (q_pred_min.mean() - q_pred.mean())).mean()
        
        critic_loss = critic_loss + cql_loss

        metrics = {
            "critic_loss": critic_loss.item(),
            "q_pred_min": q_pred_min.mean().item(),
            "q_target_next_states": q_target_next_states.mean().item(),
            "cql_loss": cql_loss.item(),
        }

        return critic_loss, metrics

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        states, actions, rewards, next_states, dones = batch

        metrics = {}

        # Alpha update
        if self.alpha_autotune:
            alpha_loss = self._alpha_loss(states)
            self.log_alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
            metrics["alpha_loss"] = alpha_loss.item()
            metrics["alpha"] = self.alpha.item()

        # Critic update
        critic_loss, critic_metrics = self._critic_loss(
            states, actions, rewards, next_states, dones
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        metrics.update(critic_metrics)

        # Actor update
        actor_loss = self._actor_loss(states)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        metrics["actor_loss"] = actor_loss.item()

        # Target network update
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return metrics

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha,
            "log_alpha_optimizer": self.log_alpha_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
        self.log_alpha = state_dict["log_alpha"]
        self.log_alpha_optimizer.load_state_dict(state_dict["log_alpha_optimizer"])
        self.total_it = state_dict["total_it"]


@torch.no_grad()
def eval_actor(
    env,
    actor: Actor,
    device: str,
    n_episodes: int,
    state_mean: Union[np.ndarray, float],
    state_std: Union[np.ndarray, float],
) -> np.ndarray:
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            state_normalized = (state - state_mean) / state_std
            action, _ = actor(torch.tensor(state_normalized, device=device).unsqueeze(0))
            state, reward, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
            done = terminated or truncated
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)


# 假设你已有的 ReplayBuffer 类如下（简化）：
class SimpleReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, size: int):
        self._states = np.zeros([size, obs_dim], dtype=np.float32)
        self._actions = np.zeros([size, action_dim], dtype=np.float32)
        self._rewards = np.zeros([size], dtype=np.float32)
        self._next_states = np.zeros([size, obs_dim], dtype=np.float32)
        self._dones = np.zeros([size], dtype=np.float32)
        self.max_size = size
        self.ptr, self.size = 0, 0

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: float,
    ):
        self._states[self.ptr] = obs
        self._next_states[self.ptr] = next_obs
        self._actions[self.ptr] = act
        self._rewards[self.ptr] = rew
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


@pyrallis.wrap()
def train(config: TrainConfig):
    set_seed(config.seed)
    
    manari_dataset = minari.load_dataset(config.env)
    print('minari dataset len is :', manari_dataset.total_steps)

    state_dim = manari_dataset.observation_space.shape[0]
    action_dim = manari_dataset.action_space.shape[0]

    buffer = SimpleReplayBuffer(state_dim, action_dim, manari_dataset.total_steps)
    for ep in manari_dataset.iterate_episodes():
        for i in range(ep.observations.shape[0] - 1):
            obs = ep.observations[i]
            next_obs = ep.observations[i+1]
            action = ep.actions[i]
            reward = ep.rewards[i]
            done = (ep.terminations | ep.truncations)[i].astype(np.float32)
            buffer.store(obs, action, reward, next_obs, done)

    dataset = {
            "observations": buffer._states,
            "actions": buffer._actions,
            "rewards": buffer._rewards,
            "next_observations": buffer._next_states,
            "terminals": buffer._dones,
        }

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
        dataset["observations"] = normalize_states(
            dataset["observations"], state_mean, state_std
        )
        dataset["next_observations"] = normalize_states(
            dataset["next_observations"], state_mean, state_std
        )
    else:
        state_mean, state_std = 0, 1

    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        manari_dataset.total_steps,
        config.device,
    )
    replay_buffer.load_minari_dataset(dataset, normalize_reward=config.normalize_reward)

    env = gym.make('Hopper-v5', ctrl_cost_weight=1e-3)

    actor = Actor(state_dim, action_dim, config.hidden_dim).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_learning_rate)
    critic = VectorizedCritic(state_dim, action_dim, config.hidden_dim, config.num_critics).to(config.device)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.critic_learning_rate)

    trainer = ContinuousCQL(
        actor=actor,
        actor_optimizer=actor_optimizer,
        critic=critic,
        critic_optimizer=critic_optimizer,
        gamma=config.discount,
        tau=config.tau,
        alpha_learning_rate=config.alpha_learning_rate,
        alpha_init=config.alpha_init,
        alpha_autotune=config.alpha_autotune,
        target_entropy=config.target_entropy,
        cql_alpha=config.cql_alpha,
        cql_n_actions=config.cql_n_actions,
        cql_log_alpha=config.cql_log_alpha,
        cql_target_action_gap=config.cql_target_action_gap,
        cql_clip_diff_min=config.cql_clip_diff_min,
        cql_clip_diff_max=config.cql_clip_diff_max,
        device=config.device,
    )

    if config.checkpoints_path:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    evaluations = []
    for t in trange(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)
        # wandb.log(log_dict, step=trainer.total_it)
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            eval_scores = eval_actor(
                env,
                actor,
                device=config.device,
                n_episodes=config.n_episodes,
                state_mean=state_mean,
                state_std=state_std,
            )
            eval_score = eval_scores.mean()
            # normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
            evaluations.append(eval_score)
            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"{eval_score:.3f} , Minari score mean: {eval_score:.3f}"
            )
            print("---------------------------------------")
            if config.checkpoints_path:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )
            # wandb.log(
            #     {"d4rl_normalized_score": normalized_eval_score},
            #     step=trainer.total_it,
            # )


if __name__ == "__main__":
    train()