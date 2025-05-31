import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# ------------------------------
# Custom Regime-Aware Environment
# ------------------------------
class RegimeAwarePortfolioEnv(gym.Env):
    def __init__(self, returns_df, regime_probs, weights_dim, horizon=20):
        super().__init__()
        self.returns_df = returns_df.reset_index(drop=True)
        self.regime_probs = regime_probs.reset_index(drop=True)
        self.weights_dim = weights_dim
        self.horizon = horizon

        self.action_space = spaces.Box(low=0, high=1, shape=(weights_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(weights_dim + regime_probs.shape[1],),
            dtype=np.float32
        )

        self.current_step = 0
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.done = False
        return self._get_observation(), {}

    def step(self, action):
        action = np.clip(action, 0, 1)
        if action.sum() == 0:
            action = np.ones_like(action) / len(action)
        else:
            action /= action.sum()

        returns = self.returns_df.iloc[self.current_step].values
        reward = np.dot(action, returns)
        reward = np.clip(reward, -0.5, 0.5)
        log_return = np.log(1 + reward)

        self.current_step += 1
        terminated = self.current_step >= self.horizon
        truncated = False
        obs = self._get_observation()
        info = {
            'weights': action,
            'regime_probs': self.regime_probs.iloc[self.current_step - 1].values
        }

        return obs, log_return, terminated, truncated, info

    def _get_observation(self):
        returns = self.returns_df.iloc[self.current_step].values
        regime = self.regime_probs.iloc[self.current_step].values
        obs = np.concatenate([returns, regime])
        return obs.astype(np.float32)

# ------------------------------
# Load Data & Train PPO
# ------------------------------
returns_df = pd.read_csv("returns_matrix.csv")
regime_probs = pd.read_csv("hmm_regime_probs.csv")
weights_dim = returns_df.shape[1]

# Create training environment
env = DummyVecEnv([lambda: RegimeAwarePortfolioEnv(returns_df, regime_probs, weights_dim, horizon=30)])

# Train the PPO agent
model = PPO("MlpPolicy", env, verbose=1, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99)
model.learn(total_timesteps=100_000)

# Save the trained model
model.save("ppo_regime_portfolio")
