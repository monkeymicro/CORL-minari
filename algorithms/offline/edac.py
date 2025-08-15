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

import minari
import gymnasium as gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributions import Normal
from tqdm import trange

@dataclass
class TrainConfig:
    # wandb params
    project: str = "CORL"
    group: str = "EDAC-MINARI"
    name: str = "EDAC"
    # model params
    hidden_dim: int = 256
    num_critics: int = 10
    gamma: float = 0.99
    tau: float = 5e-3
    eta: float = 1.0
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    max_action: float = 1.0
    # training params
    buffer_size: int = 1_000_000
    env_name: str = 'mujoco/hopper/expert-v0'
    batch_size: int = 256
    num_epochs: int = 3000
    num_updates_on_epoch: int = 1000
    normalize_reward: bool = False
    normalize: bool = False # Normalize states
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


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std

def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
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


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    def normalize_state(state):
        return (state - state_mean) / state_std

    def scale_reward(reward):
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


# SAC Actor & Critic implementation
class VectorizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, ensemble_size: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # input: [ensemble_size, batch_size, in_features]
        # output: [ensemble_size, batch_size, out_features]
        return x @ self.weight + self.bias


class VectorizedCritic(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int, num_critics: int
    ):
        super().__init__()
        self.num_critics = num_critics
        self.trunk = VectorizedLinear(state_dim + action_dim, hidden_dim, num_critics)
        self.head = VectorizedLinear(hidden_dim, 1, num_critics)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state_action = torch.cat([state, action], dim=-1)
        # [num_critics, batch_size, in_features]
        state_action = state_action.unsqueeze(0).repeat(self.num_critics, 1, 1)
        q_values = self.head(F.relu(self.trunk(state_action)))
        # [num_critics, batch_size, 1]
        return q_values


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.q_value_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.q_value_model(x)


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
        # TanhTransform()
        self.dist = Normal(0, 1)

    def forward(self, state: torch.Tensor):
        x = self.trunk(state)
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = log_std.exp()
        policy_dist = Normal(mean, std)
        return policy_dist


class EDAC:
    def __init__(
        self,
        actor: Actor,
        actor_optimizer: torch.optim.Optimizer,
        critics: VectorizedCritic,
        critic_optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha_learning_rate: float = 1e-4,
        max_action: float = 1.0,
        eta: float = 1.0,
        device: str = "cpu",
    ):
        self.device = device
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.critics = critics
        self.critic_target = deepcopy(critics)
        self.critic_optimizer = critic_optimizer

        self.log_alpha = torch.tensor([0.0], requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_learning_rate
        )

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.eta = eta
        self.target_entropy = -self.actor.dist.base_dist.mean.shape[0]

        self.total_updates = 0

    def _update_critic(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ):
        with torch.no_grad():
            next_actions = self.actor(next_state).sample()
            # next_actions = next_actions * self.max_action
            next_q_values = self.critic_target(next_state, next_actions)
            # [num_critics, batch_size, 1]
            min_next_q = torch.min(next_q_values, dim=0)[0]
            target_q = reward + (1 - done) * self.gamma * min_next_q

        q_values = self.critics(state, action)
        # [num_critics, batch_size, 1]
        critic_loss = F.mse_loss(q_values, target_q.unsqueeze(0).expand_as(q_values))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss

    def _update_actor_and_alpha(self, state: torch.Tensor):
        policy_dist = self.actor(state)
        actions = policy_dist.rsample()
        actions = torch.tanh(actions)
        log_prob = policy_dist.log_prob(torch.atanh(actions))
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        # action = action * self.max_action

        q_values = self.critics(state, actions)
        min_q = torch.min(q_values, dim=0)[0]

        actor_loss = self.alpha.detach() * log_prob.mean() - min_q.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update alpha
        alpha_loss = -self.alpha * (log_prob + self.target_entropy).detach().mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return actor_loss, alpha_loss

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def update(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_updates += 1
        state, action, reward, next_state, done = batch
        # action = action / self.max_action

        critic_loss = self._update_critic(state, action, reward, next_state, done)
        actor_loss, alpha_loss = self._update_actor_and_alpha(state)

        # Update target networks
        soft_update(self.critic_target, self.critics, self.tau)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critics": self.critics.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.item(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "total_updates": self.total_updates,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.critics.load_state_dict(state_dict["critics"])
        self.critic_target.load_state_dict(state_dict["critic_target"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
        self.log_alpha.data[0] = state_dict["log_alpha"]
        self.alpha_optimizer.load_state_dict(state_dict["alpha_optimizer"])
        self.total_updates = state_dict["total_updates"]


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
    for _ in range(n_episodes):
        state, _ = env.reset(seed=seed)
        done = False
        episode_reward = 0.0
        while not done:
            state_normalized = (state - state_mean) / state_std
            action = actor.act(torch.FloatTensor(state_normalized), device)
            state, reward, terminated, truncated, _= env.step(action)
            done = terminated or truncated
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)


def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset, env_name, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0

# 假设你已有的 ReplayBuffer 类如下（简化）：
class SimpleReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, size: int,):
        self._states = np.zeros([size, obs_dim], dtype=np.float32)
        self._actions = np.zeros([size, action_dim], dtype=np.float32)
        self._rewards = np.zeros([size], dtype=np.float32)
        self._next_states = np.zeros([size, obs_dim], dtype=np.float32)
        self._dones = np.zeros([size], dtype=np.float32)
        self.max_size = size
        self.ptr, self.size, = 0, 0
    def store(self, obs: np.ndarray,
        act: np.ndarray, 
        rew: float, 
        next_obs: np.ndarray, 
        done: float,):

        self._states[self.ptr] = obs
        self._next_states[self.ptr] = next_obs
        self._actions[self.ptr] = act
        self._rewards[self.ptr] = rew
        self._dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        # print(f"Loaded {len(data['observations'])} transitions.")

@pyrallis.wrap()
def train(config: TrainConfig):
    manari_dataset = minari.load_dataset(config.env_name)
    print('minari dataset len is :', manari_dataset.total_steps)
    # 获取维度信息
    state_dim = manari_dataset.observation_space.shape[0]
    action_dim = manari_dataset.action_space.shape[0]

    # 创建你的 ReplayBuffer
    buffer = SimpleReplayBuffer(state_dim, action_dim, manari_dataset.total_steps)

    # 使用 iterate_episodes 来提取数据
    for ep in manari_dataset.iterate_episodes():
        for i in range(ep.observations.shape[0] - 1):
            obs = ep.observations[i]
            # print(obs)
            next_obs = ep.observations[i+1]
            action = ep.actions[i]
            reward = ep.rewards[i]
            done = (ep.terminations | ep.truncations)[i].astype(np.float32)

            buffer.store(obs, action, reward, next_obs, done)

    print("Replay buffer filled.")
    
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
    replay_buffer.load_minari_dataset(dataset)
    env = gym.make('Hopper-v5', ctrl_cost_weight=1e-3)

    # Actor & Critic setup
    actor = Actor(state_dim, action_dim, config.hidden_dim, config.max_action)
    actor.to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_learning_rate)
    critic = VectorizedCritic(
        state_dim, action_dim, config.hidden_dim, config.num_critics
    )
    critic.to(config.device)
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=config.critic_learning_rate
    )

    trainer = EDAC(
        actor=actor,
        actor_optimizer=actor_optimizer,
        critic=critic,
        critic_optimizer=critic_optimizer,
        gamma=config.gamma,
        tau=config.tau,
        eta=config.eta,
        alpha_learning_rate=config.alpha_learning_rate,
        device=config.device,
    )
    # saving config to the checkpoint
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    total_updates = 0.0
    for epoch in trange(config.num_epochs, desc="Training"):
        # training
        for _ in trange(config.num_updates_on_epoch, desc="Epoch", leave=False):
            batch = replay_buffer.sample(config.batch_size)
            update_info = trainer.update(batch)

            # if total_updates % config.log_every == 0:
            #     wandb.log({"epoch": epoch, **update_info})

            total_updates += 1

        # evaluation
        if epoch % config.eval_every == 0 or epoch == config.num_epochs - 1:
            eval_scores = eval_actor(
                env=env,
                actor=actor,
                n_episodes=config.eval_episodes,
                seed=config.eval_seed,
                device=config.device,
            )
            eval_score = eval_scores.mean()
            print("---------------------------------------")
            print(
                f"Evaluation over {config.eval_episodes} episodes: "
                f"{eval_score:.3f} , Minari score mean: {eval_score:.3f}"
            )
            print("---------------------------------------")
            # wandb.log(eval_log)

            if config.checkpoints_path is not None:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"{epoch}.pt"),
                )

    # wandb.finish()


if __name__ == "__main__":
    train()
