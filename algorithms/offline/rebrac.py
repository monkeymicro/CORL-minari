# source: https://github.com/tinkoff-ai/ReBRAC
# https://arxiv.org/abs/2305.09836

import os

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # For reproducibility

import math
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Callable, Dict, Sequence, Tuple, Union

import chex
import minari  # noqa
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyrallis
import wandb
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from tqdm.auto import trange

default_kernel_init = nn.initializers.lecun_normal()
default_bias_init = nn.initializers.zeros


@dataclass
class Config:
    # wandb params
    project: str = "CORL"
    group: str = "rebrac"
    name: str = "rebrac"
    # model params
    actor_learning_rate: float = 1e-3
    critic_learning_rate: float = 1e-3
    hidden_dim: int = 256
    actor_n_hiddens: int = 3
    critic_n_hiddens: int = 3
    gamma: float = 0.99
    tau: float = 5e-3
    actor_bc_coef: float = 1.0
    critic_bc_coef: float = 1.0
    # training params
    buffer_size: int = 1_000_000
    env_name: str = "mujoco/hopper/expert-v0"
    batch_size: int = 256
    num_epochs: int = 1000
    num_updates_on_epoch: int = 500
    normalize: bool = False # Normalize states
    normalize_reward: bool = False # Normalize reward
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 5
    # general params
    checkpoints_path: Optional[str] = None
    deterministic_jax: bool = False
    train_seed: int = 10
    eval_seed: int = 42

    def __post_init__(self):
        self.name = f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def compute_mean_std(states: np.ndarray, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std

def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        normalize_reward: bool = False
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0
        
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

        self._normalize_reward = normalize_reward

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

    def sample(self, batch_size: int) -> Tuple[jnp.ndarray]:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        return (
            jnp.asarray(self.states[indices]),
            jnp.asarray(self.actions[indices]),
            jnp.asarray(self.rewards[indices]),
            jnp.asarray(self.next_states[indices]),
            jnp.asarray(self.dones[indices]),
        )


class MLP(nn.Module):
    hidden_dim: int
    n_hiddens: int
    output_dim: int
    layernorm: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i in range(self.n_hiddens):
            x = nn.Dense(
                self.hidden_dim,
                kernel_init=default_kernel_init,
                bias_init=default_bias_init,
            )(x)
            if self.layernorm:
                x = nn.LayerNorm()(x)
            x = nn.relu(x)
        x = nn.Dense(self.output_dim, kernel_init=default_kernel_init)(x)
        return x


class Actor(nn.Module):
    hidden_dim: int
    n_hiddens: int
    action_dim: int

    @nn.compact
    def __call__(
        self, state: jnp.ndarray, deterministic: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        trunk = MLP(self.hidden_dim, self.n_hiddens, self.hidden_dim)(state)
        mu = nn.Dense(
            self.action_dim, kernel_init=default_kernel_init, bias_init=default_bias_init
        )(trunk)
        log_std = nn.Dense(
            self.action_dim, kernel_init=default_kernel_init, bias_init=default_bias_init
        )(trunk)
        log_std = jnp.clip(log_std, -20.0, 2.0)
        std = jnp.exp(log_std)

        mu = jnp.clip(mu, -20.0, 20.0)

        if deterministic:
            return jnp.tanh(mu), None

        pi = mu + jax.random.normal(self.make_rng("noise"), shape=mu.shape) * std
        return jnp.tanh(pi), None


class Critic(nn.Module):
    hidden_dim: int
    n_hiddens: int
    num_critics: int = 1

    @nn.compact
    def __call__(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([state, action], axis=-1)
        q_values = MLP(
            hidden_dim=self.hidden_dim,
            n_hiddens=self.n_hiddens,
            output_dim=self.num_critics,
        )(x)
        return q_values


class ReBRAC:
    def __init__(self, rng: chex.PRNGKey, config: Config):
        self.config = config
        self.rng = rng

        self.dummy_state = jnp.zeros((1, 1), dtype=jnp.float32)
        self.dummy_action = jnp.zeros((1, 1), dtype=jnp.float32)

        self.actor = Actor(
            self.config.hidden_dim, self.config.actor_n_hiddens, 1
        )
        self.actor_state = TrainState.create(
            apply_fn=self.actor.apply,
            params=self.actor.init(
                {"params": self.rng, "noise": self.rng}, self.dummy_state
            )["params"],
            tx=optax.adam(self.config.actor_learning_rate),
        )

        self.critic = Critic(self.config.hidden_dim, self.config.critic_n_hiddens, 1)
        self.critic_state = TrainState.create(
            apply_fn=self.critic.apply,
            params=self.critic.init(self.rng, self.dummy_state, self.dummy_action)[
                "params"
            ],
            tx=optax.adam(self.config.critic_learning_rate),
        )
        self.target_critic_params = deepcopy(self.critic_state.params)

        self.log_alpha = chex.Array(0.0)
        self.log_alpha_optimizer = optax.adam(self.config.actor_learning_rate)
        self.log_alpha_state = self.log_alpha_optimizer.init(self.log_alpha)

        self.target_entropy = -1

    def update(
        self,
        rng: chex.PRNGKey,
        actor_state: TrainState,
        critic_state: TrainState,
        target_critic_params: FrozenDict,
        log_alpha_state: optax.OptState,
        log_alpha: chex.Array,
        batch: Dict[str, jnp.ndarray],
    ):
        rng, actor_key, critic_key = jax.random.split(rng, 3)

        def loss_alpha_fn(log_alpha_frozen: chex.Array, a_key: chex.PRNGKey):
            """
            Alpha loss is independent of critic.
            """
            mu, _ = self.actor.apply(
                {"params": actor_state.params},
                batch["states"],
                deterministic=False,
                rngs={"noise": a_key},
            )
            # `actions` above is sampled from a gaussian, while batch["actions"] is taken
            # from the replay buffer. `log_pi` will be negative since we are taking the
            # log probability of the sampled action under the new policy distribution
            log_pi = jnp.log(1 - jnp.tanh(mu) ** 2 + 1e-6)
            alpha = jnp.exp(log_alpha_frozen)
            alpha_loss = -alpha * (log_pi + self.target_entropy).mean()
            return alpha_loss, alpha

        def critic_loss_fn(
            critic_params: FrozenDict, target_critic_params: FrozenDict
        ):
            q_target = self.critic.apply(
                {"params": target_critic_params},
                batch["next_states"],
                batch["actions"],
            )

            q_pred = self.critic.apply(
                {"params": critic_params}, batch["states"], batch["actions"]
            )
            # Using the replay buffer actions
            # q_target = batch['rewards'] + self.config.gamma * (1-batch['dones'])*q_target
            # critic_loss = ((q_target - q_pred)**2).mean()

            # Using the replay buffer actions
            y = (
                batch["rewards"]
                + self.config.gamma * (1 - batch["dones"]) * q_target
            )
            q_loss = (q_pred - y) ** 2
            critic_loss = jnp.mean(q_loss)
            return critic_loss

        # Using the replay buffer actions for the actor loss
        def actor_loss_fn(actor_params: FrozenDict, c_key: chex.PRNGKey):
            q_pred = self.critic.apply(
                {"params": critic_state.params},
                batch["states"],
                batch["actions"],
            )
            actor_loss = -(q_pred).mean()
            return actor_loss

        # alpha update
        (alpha_loss, alpha), log_alpha_grads = jax.value_and_grad(
            loss_alpha_fn, has_aux=True
        )(log_alpha, actor_key)
        log_alpha_updates, log_alpha_state = self.log_alpha_optimizer.update(
            log_alpha_grads, log_alpha_state
        )
        log_alpha = optax.apply_updates(log_alpha, log_alpha_updates)

        # critic update
        critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(
            critic_state.params, target_critic_params
        )
        critic_state = critic_state.apply_gradients(grads=critic_grads)
        target_critic_params = optax.incremental_update(
            critic_state.params, target_critic_params, self.config.tau
        )

        # actor update
        actor_loss, actor_grads = jax.value_and_grad(actor_loss_fn)(
            actor_state.params, critic_key
        )
        actor_state = actor_state.apply_gradients(grads=actor_grads)

        metrics = {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "alpha": alpha,
        }

        return (
            rng,
            actor_state,
            critic_state,
            target_critic_params,
            log_alpha_state,
            log_alpha,
            metrics,
        )


@partial(jax.jit, static_argnames=("actor_apply_fn", "n_episodes"))
def evaluate(
    env: gym.Env,
    actor_params: FrozenDict,
    actor_apply_fn: Callable,
    n_episodes: int,
    seed: int,
    state_mean: Union[np.ndarray, float],
    state_std: Union[np.ndarray, float],
):
    def step(carry, _):
        state, key = carry
        state_normalized = (state - state_mean) / state_std
        action, _ = actor_apply_fn(
            {"params": actor_params},
            state_normalized,
            deterministic=True,
            rngs={"noise": key},
        )
        next_state, reward, terminated, truncated, _ = env.step(np.array(action))
        done = terminated or truncated
        return (
            (next_state, key),
            (reward, done),
        )

    states, _ = env.reset(seed=seed)
    keys = jax.random.split(jax.random.PRNGKey(seed), n_episodes)
    (states, _), (rewards, dones) = jax.lax.scan(
        step,
        (states, keys),
        None,
        length=env.spec.max_episode_steps,
    )

    return jnp.sum(rewards, axis=0)


@pyrallis.wrap()
def train(config: Config):
    if config.deterministic_jax:
        os.environ["JAX_ENABLE_X64"] = "1"
        chex.set_determinism(True)
    rng = jax.random.PRNGKey(config.train_seed)
    rng, dataset_rng = jax.random.split(rng)

    eval_env = gym.make(config.env_name)

    dataset = minari.load_dataset(config.env_name, download=True, seed=None)

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

    # Initializing ReBRAC agent
    rng, agent_rng = jax.random.split(rng)
    agent = ReBRAC(agent_rng, config)

    # Initializing train states
    actor_state = agent.actor_state
    critic_state = agent.critic_state
    target_critic_params = agent.target_critic_params
    log_alpha_state = agent.log_alpha_state
    log_alpha = agent.log_alpha

    # Jitting the update function
    update_fn = jax.jit(agent.update)
    actor_action_fn = jax.jit(agent.actor.apply)

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    wandb.init(
        config=asdict(config),
        project=config.project,
        group=config.group,
        name=config.name,
        id=str(uuid.uuid4()),
        save_code=True,
    )
    wandb.run.save()

    for epoch in trange(config.num_epochs, desc="Training"):
        rng, update_rng = jax.random.split(rng)

        def td3_loop_update_step(carry, _):
            (
                rng,
                actor_state,
                critic_state,
                target_critic_params,
                log_alpha_state,
                log_alpha,
                metrics,
            ) = carry
            rng, key = jax.random.split(rng)
            batch = replay_buffer.sample(config.batch_size)
            (
                rng,
                actor_state,
                critic_state,
                target_critic_params,
                log_alpha_state,
                log_alpha,
                update_metrics,
            ) = update_fn(
                key,
                actor_state,
                critic_state,
                target_critic_params,
                log_alpha_state,
                log_alpha,
                batch,
            )
            # update_metrics is a dict of individual metric values.
            # We add them to the metrics of the current epoch to compute mean.
            # metrics = metrics.update(update_metrics)
            return (
                rng,
                actor_state,
                critic_state,
                target_critic_params,
                log_alpha_state,
                log_alpha,
                update_metrics,
            ), None

        update_carry = (
            update_rng,
            actor_state,
            critic_state,
            target_critic_params,
            log_alpha_state,
            log_alpha,
            None,
        )

        update_carry, _ = jax.lax.scan(
            f=lambda carry, _: td3_loop_update_step(carry, _),
            init=update_carry,
            xs=None,
            length=config.num_updates_on_epoch,
        )

        metrics_of_epoch = update_carry[6]
        wandb.log(
            {
                "epoch": epoch,
                "actor_loss": metrics_of_epoch["actor_loss"].mean(),
                "critic_loss": metrics_of_epoch["critic_loss"].mean(),
                "alpha_loss": metrics_of_epoch["alpha_loss"].mean(),
                "alpha": metrics_of_epoch["alpha"].mean(),
            }
        )
        if epoch % config.eval_every == 0 or epoch == config.num_epochs - 1:
            eval_returns = evaluate(
                eval_env,
                update_carry[1].params,
                actor_action_fn,
                config.eval_episodes,
                seed=config.eval_seed,
                state_mean=state_mean,
                state_std=state_std,
            )

            wandb.log(
                {
                    "epoch": epoch,
                    "eval/return_mean": np.mean(eval_returns),
                    "eval/return_std": np.std(eval_returns),
                }
            )

            eval_score = eval_returns.mean()
            print("---------------------------------------")
            print(
                f"Evaluation over {config.eval_episodes} episodes: "
                f"{eval_score:.3f} , Minari score mean: {eval_score:.3f}"
            )
            print("---------------------------------------")
            if config.checkpoints_path is not None:
                frozen_dict = FrozenDict(
                    {
                        "actor_state": update_carry[1],
                        "critic_state": update_carry[2],
                        "target_critic_params": update_carry[3],
                        "log_alpha_state": update_carry[4],
                        "log_alpha": update_carry[5],
                        "state_mean": state_mean,
                        "state_std": state_std,
                    }
                )
                with open(
                    os.path.join(config.checkpoints_path, f"{epoch}.pickle"), "wb"
                ) as f:
                    import pickle

                    pickle.dump(frozen_dict, f)


if __name__ == "__main__":
    train()