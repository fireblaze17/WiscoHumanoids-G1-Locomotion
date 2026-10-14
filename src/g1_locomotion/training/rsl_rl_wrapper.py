"""RSL-RL bridge with bounded actions and complete per-episode resets."""

from __future__ import annotations

from collections import deque
import functools

import jax

from mujoco_playground._src import wrapper
from mujoco_playground._src import wrapper_torch


class G1RSLRLBraxWrapper(wrapper_torch.RSLRLBraxWrapper):
    """Uses Playground's Torch bridge with action clipping and full resets."""

    def __init__(
        self,
        env,
        num_actors,
        seed,
        episode_length,
        action_repeat,
        randomization_fn=None,
        render_callback=None,
        device_rank=None,
    ):
        self.seed = seed
        self.batch_size = num_actors
        self.num_envs = num_actors
        self.key = jax.random.PRNGKey(self.seed)

        if device_rank is not None:
            gpu_devices = jax.devices("gpu")
            self.key = jax.device_put(self.key, gpu_devices[device_rank])
            self.device = f"cuda:{device_rank}"
            print(f"Device -- {gpu_devices[device_rank]}")
            print(f"Key device -- {self.key.devices()}")

        key_reset, key_randomization = jax.random.split(self.key)
        self.key_reset = jax.random.split(key_reset, self.batch_size)

        if randomization_fn is not None:
            randomization_rng = jax.random.split(key_randomization, self.batch_size)
            vectorized_randomization_fn = functools.partial(
                randomization_fn, rng=randomization_rng
            )
        else:
            vectorized_randomization_fn = None

        self.env = wrapper.wrap_for_brax_training(
            env,
            episode_length=episode_length,
            action_repeat=action_repeat,
            randomization_fn=vectorized_randomization_fn,
            full_reset=True,
        )
        self.render_callback = render_callback

        obs_shape = self.env.env.unwrapped.observation_size
        self.asymmetric_obs = isinstance(obs_shape, dict)
        print(f"obs_shape: {obs_shape}")
        if self.asymmetric_obs:
            print("Asymmetric observation space")
            self.num_obs = obs_shape["state"]
            self.num_privileged_obs = obs_shape["privileged_state"]
        else:
            self.num_obs = obs_shape
            self.num_privileged_obs = None

        self.num_actions = self.env.env.unwrapped.action_size
        self.max_episode_length = episode_length
        self.success_queue = deque(maxlen=100)

        print("JITing reset and step")
        self.reset_fn = jax.jit(self.env.reset)
        self.step_fn = jax.jit(self.env.step)
        print("Done JITing reset and step")
        self.env_state = None

    def step(self, action):
        """Logs raw policy output, then delegates clipping and stepping upstream."""
        raw_abs = action.detach().abs()
        raw_metrics = {
            "action/raw_abs_mean": raw_abs.mean().item(),
            "action/raw_abs_max": raw_abs.max().item(),
            "action/clip_fraction": (raw_abs > 1.0).float().mean().item(),
        }
        obs, reward, done, info = super().step(action)
        info["log"].update(raw_metrics)
        return obs, reward, done, info
