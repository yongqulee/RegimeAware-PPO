import numpy as np
import pandas as pd
import gym
from gym import spaces

class RegimeAwarePortfolioEnv(gym.Env):
    def __init__(self, returns_df, regime_probs, weights_dim, horizon=20):
        super().__init__()
        self.returns_df = returns_df.reset_index(drop=True)
        self.regime_probs = regime_probs.reset_index(drop=True)
        self.weights_dim = weights_dim
        self.horizon = horizon

        self.action_space = spaces.Box(low=0, high=1, shape=(weights_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(weights_dim + 3,), dtype=np.float32)

        self.current_step = 0
        self.done = False

    def reset(self):
        self.current_step = 0
        self.done = False
        self.weights = np.ones(self.weights_dim) / self.weights_dim
        return self._get_observation()

    def step(self, action):
        action = np.clip(action, 0, 1)
        action /= action.sum()
        returns = self.returns_df.iloc[self.current_step].values
        reward = np.dot(action, returns)

        self.current_step += 1
        self.done = self.current_step >= self.horizon
        obs = self._get_observation()
        info = {'weights': action, 'regime_probs': self.regime_probs.iloc[self.current_step - 1].values}
        return obs, reward, self.done, info

    def _get_observation(self):
        returns = self.returns_df.iloc[self.current_step].values
        regime = self.regime_probs.iloc[self.current_step].values
        obs = np.concatenate([returns, regime])
        return obs.astype(np.float32)
