
import numpy as np
import pandas as pd
import gym
from gym import spaces

class RegimeAwarePortfolioEnvTransformer(gym.Env):
    def __init__(self, returns_df, regime_preds, weights_dim, horizon=30, cost_rate=0.001,
                 reset_interval=30, shock_interval=25, shock_magnitude=-0.05):
        super().__init__()
        self.returns_df = returns_df.reset_index(drop=True)
        self.regime_preds = regime_preds.reset_index(drop=True)
        self.weights_dim = weights_dim
        self.horizon = horizon
        self.cost_rate = cost_rate
        self.reset_interval = reset_interval
        self.shock_interval = shock_interval
        self.shock_magnitude = shock_magnitude

        self.action_space = spaces.Box(low=0, high=1, shape=(weights_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(weights_dim + regime_preds.shape[1],), dtype=np.float32)

        self.current_step = 0
        self.prev_action = np.ones(weights_dim) / weights_dim
        self.total_return = 1.0

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.prev_action = np.ones(self.weights_dim) / self.weights_dim
        self.total_return = 1.0
        return self._get_observation(), {}

    def step(self, action):
        action = np.clip(action, 0, 1)
        action /= action.sum() if action.sum() > 0 else 1.0

        returns = self.returns_df.iloc[self.current_step].values
        reward = np.dot(action, returns)

        window = max(0, self.current_step - 5)
        past_returns = self.returns_df.iloc[window:self.current_step+1].dot(action)
        volatility = past_returns.std() if len(past_returns) > 1 else 1e-6
        reward /= max(volatility, 1e-2)

        transaction_cost = np.sum(np.abs(action - self.prev_action)) * self.cost_rate
        reward -= transaction_cost
        self.prev_action = action

        reward = np.clip(reward, -0.03, 0.03)
        log_return = np.log(1 + reward)

        if self.current_step > 0 and self.current_step % self.shock_interval == 0:
            log_return += self.shock_magnitude

        self.total_return *= np.exp(log_return)
        if self.current_step > 0 and self.current_step % self.reset_interval == 0:
            self.total_return = 1.0

        self.current_step += 1
        terminated = self.current_step >= self.horizon
        truncated = False
        info = {
            "weights": action,
            "volatility": volatility,
            "transaction_cost": transaction_cost,
            "total_return": self.total_return,
            "reward": reward
        }

        return self._get_observation(), log_return, terminated, truncated, info

    def _get_observation(self):
        returns = self.returns_df.iloc[self.current_step].values
        regime = self.regime_preds.iloc[self.current_step].values
        return np.concatenate([returns, regime]).astype(np.float32)
