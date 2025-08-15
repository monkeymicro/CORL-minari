# source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf
import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import minari
import gymnasium as gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import trange

TensorBatch = List[torch.Tensor]


EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


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
    # IQL
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    tau: float = 0.005  # Target network update rate
    beta: float = 3.0  # Inverse temperature. Small beta -> BC, big beta -> maximizing Q
    iql_tau: float = 0.7  # Coefficient for asymmetric loss
    iql_deterministic: bool = False  # Use deterministic actor
    normalize: bool = False  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    vf_lr: float = 3e-4  # V function learning rate
    qf_lr: float = 3e-4  # Critic learning rate
    actor_lr: float = 3e-4  # Actor learning rate
    actor_dropout: Optional[float] = None  # Adroit uses dropout for policy network
    # Wandb logging
    project: str = "CORL"
    group: str = "IQL-D4RL"
    name: str = "IQL"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


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


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
        normalize_reward: bool = False,
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


# IQL Actor & Critic & Value Function
class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(self.dim)


class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation_fn=nn.ReLU,
        output_activation_fn=None,
        squeeze_output=False,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.dims = dims
        self.activation_fn = activation_fn
        self.output_activation_fn = output_activation_fn
        self.squeeze_output = squeeze_output
        self.dropout = dropout

        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())
            if dropout is not None:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            layers.append(Squeeze())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, dropout: Optional[float]):
        super().__init__()
        self.net = MLP(
            [state_dim, hidden_dim, hidden_dim, action_dim],
            squeeze_output=False,
            dropout=dropout
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim, requires_grad=True))

    def forward(self, state: torch.Tensor) -> Normal:
        mean = self.net(state)
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        return Normal(mean, torch.exp(log_std))


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = MLP(
            [state_dim + action_dim, hidden_dim, hidden_dim, 1], squeeze_output=True
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.net = MLP([state_dim, hidden_dim, hidden_dim, 1], squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class ImplicitQLearning:
    def __init__(
        self,
        max_timesteps: int,
        beta: float,
        iql_tau: float,
        discount: float,
        iql_deterministic: bool,
        actor_lr: float,
        qf_lr: float,
        vf_lr: float,
        actor: Actor,
        actor_optimizer: torch.optim.Optimizer,
        q_network_1: Critic,
        q_network_2: Critic,
        qf_optimizer: torch.optim.Optimizer,
        v_network: ValueFunction,
        vf_optimizer: torch.optim.Optimizer,
        device: str,
    ):
        self.beta = beta
        self.iql_tau = iql_tau
        self.discount = discount
        self.iql_deterministic = iql_deterministic
        self.device = device

        self.v_network = v_network
        self.v_optimizer = vf_optimizer
        self.q_network_1 = q_network_1
        self.q_network_2 = q_network_2
        self.q_target_1 = copy.deepcopy(self.q_network_1)
        self.q_target_2 = copy.deepcopy(self.q_network_2)
        self.q_optimizer = qf_optimizer
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.actor_lr_scheduler = CosineAnnealingLR(
            self.actor_optimizer, max_timesteps
        )
        self.total_it = 0

    def _train_value_function(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            q_target = torch.min(
                self.q_target_1(states, actions), self.q_target_2(states, actions)
            )

        v = self.v_network(states)
        vf_loss = F.mse_loss(v, q_target)
        td_error = q_target - v
        self.v_optimizer.zero_grad()
        vf_loss.backward()
        self.v_optimizer.step()

        return td_error, vf_loss

    def _train_q_function(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_v = self.v_network(next_states)
            target_q = rewards + self.discount * (1 - dones) * next_v

        q_1 = self.q_network_1(states, actions)
        q_2 = self.q_network_2(states, actions)
        qf_loss = F.mse_loss(q_1, target_q) + F.mse_loss(q_2, target_q)

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        return qf_loss

    def _train_actor(self, states, actions, td_error):
        exp_adv = torch.exp(self.beta * td_error.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.actor(states)
        if self.iql_deterministic:
            actor_loss = F.mse_loss(policy_out.mean, actions)
        else:
            log_probs = policy_out.log_prob(actions).sum(-1, keepdim=True)
            actor_loss = (-exp_adv * log_probs).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_scheduler.step()

        return actor_loss

    def train(self, batch) -> Dict[str, float]:
        self.total_it += 1
        states, actions, rewards, next_states, dones = batch

        # Update value function
        td_error, vf_loss = self._train_value_function(
            states, actions, rewards, next_states, dones
        )
        # Update Q function
        qf_loss = self._train_q_function(
            states, actions, rewards, next_states, dones
        )
        # Update actor
        actor_loss = self._train_actor(states, actions, td_error)

        # Update target networks
        soft_update(self.q_target_1, self.q_network_1, self.iql_tau)
        soft_update(self.q_target_2, self.q_network_2, self.iql_tau)

        return {
            "qf_loss": qf_loss.item(),
            "vf_loss": vf_loss.item(),
            "actor_loss": actor_loss.item(),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "q_network_1": self.q_network_1.state_dict(),
            "q_network_2": self.q_network_2.state_dict(),
            "q_target_1": self.q_target_1.state_dict(),
            "q_target_2": self.q_target_2.state_dict(),
            "v_network": self.v_network.state_dict(),
            "actor": self.actor.state_dict(),
            "qf_optimizer": self.q_optimizer.state_dict(),
            "vf_optimizer": self.v_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_lr_scheduler": self.actor_lr_scheduler.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.q_network_1.load_state_dict(state_dict["q_network_1"])
        self.q_network_2.load_state_dict(state_dict["q_network_2"])
        self.q_target_1.load_state_dict(state_dict["q_target_1"])
        self.q_target_2.load_state_dict(state_dict["q_target_2"])
        self.v_network.load_state_dict(state_dict["v_network"])
        self.actor.load_state_dict(state_dict["actor"])
        self.q_optimizer.load_state_dict(state_dict["qf_optimizer"])
        self.v_optimizer.load_state_dict(state_dict["vf_optimizer"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_lr_scheduler.load_state_dict(state_dict["actor_lr_scheduler"])
        self.total_it = state_dict["total_it"]


@pyrallis.wrap()
def train(config: TrainConfig):
    print(config)
    # set_seed(config.seed)
    env = gym.make(config.env)

    # Note: `seed=None` to ensure random behavior in case seed is set to a non-None value in the config.
    dataset = minari.load_dataset(config.env, seed=None)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

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
        config.normalize_reward,
    )
    replay_buffer.load_minari_dataset(dataset)

    # Set up networks
    actor = Actor(state_dim, action_dim, 256, config.actor_dropout).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
    q_network_1 = Critic(state_dim, action_dim, 256).to(config.device)
    q_network_2 = Critic(state_dim, action_dim, 256).to(config.device)
    qf_optimizer = torch.optim.Adam(
        list(q_network_1.parameters()) + list(q_network_2.parameters()), lr=config.qf_lr
    )
    v_network = ValueFunction(state_dim, 256).to(config.device)
    vf_optimizer = torch.optim.Adam(v_network.parameters(), lr=config.vf_lr)

    kwargs = {
        "max_timesteps": config.max_timesteps,
        "beta": config.beta,
        "iql_tau": config.iql_tau,
        "discount": config.discount,
        "iql_deterministic": config.iql_deterministic,
        "actor_lr": config.actor_lr,
        "qf_lr": config.qf_lr,
        "vf_lr": config.vf_lr,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "q_network_1": q_network_1,
        "q_network_2": q_network_2,
        "qf_optimizer": qf_optimizer,
        "v_network": v_network,
        "vf_optimizer": vf_optimizer,
        "device": config.device,
    }

    print("---------------------------------------")
    print(f"Training IQL, Env: {config.env}")
    print("---------------------------------------")

    # Initialize actor
    trainer = ImplicitQLearning(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    wandb_init(asdict(config))

    evaluations = []
    for t in trange(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)
        wandb.log(log_dict, step=trainer.total_it)
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            eval_scores = eval_actor(
                env,
                actor,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
                state_mean=state_mean,
                state_std=state_std,
            )
            eval_score = eval_scores.mean()
            # normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
            # evaluations.append(normalized_eval_score)
            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"{eval_score:.3f} , D4RL score: {eval_score:.3f}"
            )
            print("---------------------------------------")

            if config.checkpoints_path is not None:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )

            wandb.log(
                {"d4rl_normalized_score": eval_score},
                step=trainer.total_it,
            )

@torch.no_grad()
def eval_actor(
    env,
    actor: Actor,
    device: str,
    n_episodes: int,
    seed: int,
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
            action = actor(torch.tensor(state_normalized, device=device)).sample()
            state, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
            done = terminated or truncated
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)


if __name__ == "__main__":
    train()