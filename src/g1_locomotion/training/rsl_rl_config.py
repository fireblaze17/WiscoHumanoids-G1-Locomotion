"""RSL-RL PPO configuration for the stochastic G1 flat environment."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from ml_collections import config_dict


def default_ppo_config() -> config_dict.ConfigDict:
    """Returns conservative MuJoCo Playground-style RSL-RL PPO defaults."""
    cfg = config_dict.ConfigDict()
    cfg.seed = 1
    cfg.runner_class_name = "OnPolicyRunner"
    cfg.num_steps_per_env = 24
    cfg.max_iterations = 2
    cfg.save_interval = 1
    cfg.empirical_normalization = True
    cfg.experiment_name = "g1_flat"
    cfg.run_name = ""
    cfg.resume = False
    cfg.load_run = ""
    cfg.checkpoint = -1
    cfg.logger = "tensorboard"

    cfg.policy = config_dict.ConfigDict()
    cfg.policy.class_name = "ActorCritic"
    cfg.policy.init_noise_std = 1.0
    cfg.policy.actor_hidden_dims = [512, 256, 128]
    cfg.policy.critic_hidden_dims = [512, 256, 128]
    cfg.policy.activation = "elu"

    cfg.algorithm = config_dict.ConfigDict()
    cfg.algorithm.class_name = "PPO"
    cfg.algorithm.value_loss_coef = 1.0
    cfg.algorithm.use_clipped_value_loss = True
    cfg.algorithm.clip_param = 0.2
    cfg.algorithm.entropy_coef = 0.001
    cfg.algorithm.num_learning_epochs = 5
    cfg.algorithm.num_mini_batches = 4
    cfg.algorithm.learning_rate = 3.0e-4
    cfg.algorithm.schedule = "adaptive"
    cfg.algorithm.gamma = 0.99
    cfg.algorithm.lam = 0.95
    cfg.algorithm.desired_kl = 0.01
    cfg.algorithm.max_grad_norm = 1.0
    cfg.algorithm.rnd_cfg = None
    cfg.algorithm.symmetry_cfg = None

    cfg.obs_groups = config_dict.ConfigDict()
    cfg.obs_groups.policy = ["state"]
    cfg.obs_groups.critic = ["privileged_state"]
    return cfg


def to_plain_dict(value: Any) -> Any:
    """Converts ConfigDict/list containers to JSON/YAML-friendly objects."""
    if isinstance(value, config_dict.ConfigDict):
        return {key: to_plain_dict(value[key]) for key in value}
    if isinstance(value, dict):
        return {key: to_plain_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain_dict(item) for item in value]
    if isinstance(value, tuple):
        return [to_plain_dict(item) for item in value]
    return deepcopy(value)
