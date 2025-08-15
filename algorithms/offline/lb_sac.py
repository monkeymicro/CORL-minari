# Inspired by:
# 1. paper for LB-SAC: https://arxiv.org/abs/2211.11092
# 2. implementation: https://github.com/tinkoff-ai/lb-sac
import math
import os
import random
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import minari
import gymnasium as gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import wandb
from torch.distributions import Normal
from tqdm import trange

# base batch size: 256
# base learning rate: 3e-4
@dataclass
class TrainConfig:
    # wandb params
    project: str = "CORL"
    group: str = "LB-SAC"
    name: str = "LB-SAC"
    # model params
    hidden_dim: int = 256
    num_critics: int = 10
    gamma: float = 0.99
    tau: float = 5e-3
    actor_learning_rate: float = 0.0018
    critic_learning_rate: float = 0.0018
    alpha_learning_rate: float = 0.0018
    critic_layernorm: bool = False
    edac_init: bool = False
    max_action: float = 1.0
    # training params
    buffer_size: int = 1_000_000
    env_name: str = 'mujoco/hopper/expert-v0'
    batch_size: int = 10_000
    num_epochs: int = 300
    num_updates_on_epoch: int = 1000
    normalize: bool = False  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 5
    # general params
    checkpoints_path: Optional[str] = None
    deterministic_torch: bool = False
    train_seed: int = 10
    eval_seed: int = 42
    log_every: int = 100
    device: str = "cpu"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


# general utils
TensorBatch = List[torch.Tensor]


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
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


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
        save_code=True,
    )
    wandb.run.save()


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
        normalize_reward: bool = False
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
        self._normalize_reward = normalize_reward

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in minari, i.e. from Dict[str, np.array].
    def load_minari_dataset(self, data: Dict[str, np.ndarray]):
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

        if self._normalize_reward:
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
        raise NotImplementedError


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super(Actor, self).__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(state)
        mu = self.mu_head(hidden)
        log_std = self.log_std_head(hidden)
        log_std = torch.clamp(log_std, -20, 2)
        return mu, log_std


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, layernorm: bool = False):
        super(Critic, self).__init__()
        self.q_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.q_net(x).squeeze(-1)


class EnsembleCritic(nn.Module):
    def __init__(self, num_critics: int, state_dim: int, action_dim: int, hidden_dim: int, edac_init: bool = False,
                 layernorm: bool = False):
        super().__init__()
        self.critics = nn.ModuleList([
            Critic(state_dim, action_dim, hidden_dim, layernorm)
            for _ in range(num_critics)
        ])
        if edac_init:
            for critic in self.critics:
                critic.q_net[-1].weight.data.uniform_(-3e-3, 3e-3)
                critic.q_net[-1].bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        qs = [critic(state, action) for critic in self.critics]
        return torch.stack(qs, dim=0)


class LBSAC:
    def __init__(self,
                 actor: nn.Module,
                 actor_optimizer: torch.optim.Optimizer,
                 critic: EnsembleCritic,
                 critic_optimizer: torch.optim.Optimizer,
                 target_critic: EnsembleCritic,
                 log_alpha: torch.Tensor,
                 alpha_optimizer: torch.optim.Optimizer,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 max_action: float = 1.0,
                 device: str = "cpu"):

        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action
        self.device = device

        self.actor = actor
        self.actor_optimizer = actor_optimizer

        self.critic = critic
        self.critic_optimizer = critic_optimizer
        self.target_critic = target_critic

        self.log_alpha = log_alpha
        self.alpha_optimizer = alpha_optimizer

    def update(self, batch: TensorBatch) -> Dict[str, float]:
        states, actions, rewards, next_states, dones = batch

        # --- Update critic ---
        with torch.no_grad():
            mu, log_std = self.actor(next_states)
            policy_dist = Normal(mu, torch.exp(log_std))
            next_actions = policy_dist.sample()
            next_actions = torch.tanh(next_actions) * self.max_action
            log_prob = policy_dist.log_prob(torch.atanh(next_actions / self.max_action + 1e-6)).sum(
                axis=-1) - torch.log(1 - next_actions.pow(2) + 1e-6).sum(axis=-1)
            target_q = self.target_critic(next_states, next_actions)
            min_target_q = target_q.min(0)[0].squeeze(-1)
            target_q = (rewards + self.gamma * (1 - dones.squeeze(-1)) * min_target_q)
        q_predictions = self.critic(states, actions)
        critic_loss = F.mse_loss(q_predictions, target_q.unsqueeze(0).repeat(self.critic.num_critics, 1), reduction="mean")
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        soft_update(self.target_critic, self.critic, self.tau)

        # --- Update actor ---
        mu, log_std = self.actor(states)
        policy_dist = Normal(mu, torch.exp(log_std))
        actions_pi = policy_dist.sample()
        actions_pi = torch.tanh(actions_pi) * self.max_action
        with torch.no_grad():
            q_values = self.critic(states, actions_pi)
            min_q_values = q_values.min(0)[0]
        actor_loss = ((self.alpha.detach() * policy_dist.log_prob(actions_pi).sum(axis=-1) - min_q_values).mean())
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Update alpha ---
        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item()
        }

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict()
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.target_critic.load_state_dict(state_dict["target_critic"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
        self.log_alpha.load_state_dict(state_dict["log_alpha"])
        self.alpha_optimizer.load_state_dict(state_dict["alpha_optimizer"])


@pyrallis.wrap()
def train(config: TrainConfig):
    set_seed(config.train_seed, deterministic_torch=config.deterministic_torch)
    eval_env = gym.make(config.env_name)
    eval_env.seed(config.eval_seed)
    
    dataset = minari.load_dataset(config.env_name, seed=None)
    state_dim = dataset.observations.shape[1]
    action_dim = dataset.actions.shape[1]

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset.observations, eps=1e-3)
        dataset.observations = normalize_states(dataset.observations, state_mean, state_std)
    else:
        state_mean, state_std = 0, 1

    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
        config.normalize_reward
    )
    replay_buffer.load_minari_dataset(dataset)

    actor = Actor(state_dim, action_dim, config.hidden_dim).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_learning_rate)

    critic = EnsembleCritic(config.num_critics, state_dim, action_dim, config.hidden_dim, config.edac_init, config.critic_layernorm).to(config.device)
    target_critic = deepcopy(critic).to(config.device)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.critic_learning_rate)

    log_alpha = torch.zeros(1, requires_grad=True, device=config.device)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=config.alpha_learning_rate)
    
    kwargs = {
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "critic": critic,
        "critic_optimizer": critic_optimizer,
        "target_critic": target_critic,
        "log_alpha": log_alpha,
        "alpha_optimizer": alpha_optimizer,
        "gamma": config.gamma,
        "tau": config.tau,
        "max_action": config.max_action,
        "device": config.device
    }

    trainer = LBSAC(**kwargs)
    
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    wandb_init(asdict(config))

    total_updates = 0.0
    evaluations = []
    for epoch in trange(config.num_epochs, desc="Training"):
        # training
        for _ in trange(config.num_updates_on_epoch, desc="Epoch", leave=False):
            batch = replay_buffer.sample(config.batch_size)
            batch = [b.to(config.device) for b in batch]
            update_info = trainer.update(batch)

            total_updates += 1

            if total_updates % config.log_every == 0:
                wandb.log({"epoch": epoch, **update_info}, step=total_updates)

        # evaluation
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
            eval_log = {
                "eval/reward_mean": np.mean(eval_returns),
                "eval/reward_std": np.std(eval_returns),
                "epoch": epoch,
            }
            if hasattr(eval_env, "get_normalized_score"):
                normalized_score = eval_env.get_normalized_score(eval_returns) * 100.0
                eval_log["eval/normalized_score_mean"] = np.mean(normalized_score)
                eval_log["eval/normalized_score_std"] = np.std(normalized_score)

            wandb.log(eval_log, step=total_updates)
            
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


@torch.no_grad()
def eval_actor(
    env: gym.Env,
    actor: Actor,
    n_episodes: int,
    seed: int,
    device: str,
    state_mean: Union[np.ndarray, float],
    state_std: Union[np.ndarray, float],
) -> np.ndarray:
    actor.eval()
    episode_rewards = []
    for i in range(n_episodes):
        state, _ = env.reset(seed=seed + i)
        done = False
        episode_reward = 0.0
        while not done:
            state_normalized = (state - state_mean) / state_std
            mu, _ = actor(torch.tensor(state_normalized, device=device))
            action = torch.tanh(mu).cpu().numpy()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)


if __name__ == '__main__':
    train()