"""Tactical policies sharing the five-action HighwayDecisionEnv interface."""

import numpy as np

from carla_env import (
    ACCELERATE,
    CHANGE_LEFT,
    CHANGE_RIGHT,
    DECELERATE,
    MAINTAIN,
)


class RandomPolicy:
    """Uniform seeded policy; environment-side checks handle unsafe requests."""

    version = "random-v1"

    def __init__(self, seed=None):
        self.reset(seed)

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def act(self, observation, state):
        return int(self.rng.integers(0, 5))


class KeepLanePolicy:
    """Fixed tactical policy; shared car following supplies longitudinal safety."""

    version = "keep-lane-v1"

    def reset(self, seed=None):
        return None

    def act(self, observation, state):
        return MAINTAIN


class RuleBasedOvertakingPolicy:
    """Small, auditable overtaking policy using the structured environment state."""

    version = "rule-based-v1"

    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self, seed=None):
        self.steps_since_change = 1000000

    def _lane_safe(self, lane):
        if not lane.get("available", False):
            return False
        if lane["front_gap_m"] < self.config["safe_front_gap_m"]:
            return False
        if lane["rear_gap_m"] < self.config["safe_rear_gap_m"]:
            return False
        rear_closing_speed = max(0.0, lane["rear_relative_speed_mps"])
        if rear_closing_speed > 0.0:
            rear_ttc = lane["rear_gap_m"] / rear_closing_speed
            if rear_ttc < self.config["safe_rear_ttc_seconds"]:
                return False
        return True

    def act(self, observation, state):
        self.steps_since_change += 1
        if state.get("lane_change_active", False):
            return MAINTAIN

        current = state["lanes"]["current"]
        blocked = (
            current["front_gap_m"] < self.config["blocked_front_gap_m"]
            and current["front_relative_speed_mps"]
            < self.config["blocked_relative_speed_mps"]
        )
        if blocked:
            if self._lane_safe(state["lanes"]["left"]):
                self.steps_since_change = 0
                return CHANGE_LEFT
            if self._lane_safe(state["lanes"]["right"]):
                self.steps_since_change = 0
                return CHANGE_RIGHT
            return DECELERATE

        cooldown = self.config.get("return_right_cooldown_steps", 8)
        right = state["lanes"]["right"]
        if (
            state.get("left_of_initial_lane", False)
            and self.steps_since_change >= cooldown
            and self._lane_safe(right)
            and current["front_gap_m"] >= self.config["blocked_front_gap_m"]
        ):
            self.steps_since_change = 0
            return CHANGE_RIGHT

        if state["target_speed_kmh"] < self.config["cruise_speed_kmh"]:
            return ACCELERATE
        return MAINTAIN


class PPOPolicyAdapter:
    """Deterministic Stable-Baselines3 PPO evaluation adapter."""

    version = "sb3-ppo"

    def __init__(self, model_path, env):
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise RuntimeError(
                "Stable-Baselines3 is required to load a PPO policy. "
                "Install requirements.txt first."
            ) from exc
        self.model_path = str(model_path)
        self.model = PPO.load(self.model_path, device="cpu")
        if self.model.observation_space != env.observation_space:
            raise ValueError(
                "Loaded PPO observation space does not match HighwayDecisionEnv."
            )
        if self.model.action_space != env.action_space:
            raise ValueError(
                "Loaded PPO action space does not match HighwayDecisionEnv."
            )

    def reset(self, seed=None):
        return None

    def act(self, observation, state):
        action, _ = self.model.predict(observation, deterministic=True)
        return int(np.asarray(action).item())


def build_policy(name, config, env, seed=None, model_path=None):
    """Construct exactly one of the four requested tactical policies."""

    if name == "random":
        return RandomPolicy(seed)
    if name == "keep_lane":
        return KeepLanePolicy()
    if name == "rule_based":
        return RuleBasedOvertakingPolicy(config["rule_based"])
    if name == "ppo":
        if not model_path:
            raise ValueError("The PPO policy requires --model.")
        return PPOPolicyAdapter(model_path, env)
    raise ValueError(
        "Unknown policy %r; choose ppo, random, keep_lane, or rule_based." % name
    )
