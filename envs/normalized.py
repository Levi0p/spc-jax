import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box


class NormalizedBoxEnv(Env):
    """
    Normalize action to in [-1, 1].

    """
    def __init__(
            self,
            env,
            reward_scale=1.0,
    ):
        self._wrapped_env = env
        self._reward_scale = reward_scale
        ub = np.ones(self._wrapped_env.action_space.shape)
        self.action_space = Box(-1 * ub, ub)

    def step(self, action):
        lb = self._wrapped_env.action_space.low
        ub = self._wrapped_env.action_space.high
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)

        wrapped_step = self._wrapped_env.step(scaled_action)
        next_obs, reward, terminated, truncated, info = wrapped_step
        return next_obs, reward*self._reward_scale, terminated, truncated, info

    def __str__(self):
        return "Normalized: %s" % self._wrapped_env

    def __getattr__(self, attrname):
        return getattr(self._wrapped_env, attrname)

    def reset(self, *args, **kwargs):
        return self._wrapped_env.reset(*args, **kwargs)
    
    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)
