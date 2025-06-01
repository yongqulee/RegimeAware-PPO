import shap
import torch
import numpy as np
import pandas as pd
from torch import nn
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium import Env, spaces

# === Custom Regime-Aware PPO Environment ===
class RegimeAwarePortfolioEnv(Env):
    def __init__(self, returns_df, regime_probs, weights_dim,
                 horizon=30, cost_rate=0.001,
                 reset_interval=30, shock_interval=25,
                 shock_magnitude=-0.05,
                 clip_rewards=True, apply_cost=True, reset_capital=True):
        super().__init__()
        self.returns_df = returns_df.reset_index(drop=True)
        self.regime_probs = regime_probs.reset_index(drop=True)
        self.weights_dim = weights_dim
        self.horizon = horizon
        self.cost_rate = cost_rate
        self.reset_interval = reset_interval
        self.shock_interval = shock_interval
        self.shock_magnitude = shock_magnitude
        self.clip_rewards = clip_rewards
        self.apply_cost = apply_cost
        self.reset_capital = reset_capital

        self.action_space = spaces.Box(low=0, high=1, shape=(weights_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(weights_dim + regime_probs.shape[1],), dtype=np.float32)
        self.current_step = 0
        self.prev_action = np.ones(weights_dim) / weights_dim
        self.total_return = 1.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.prev_action = np.ones(self.weights_dim) / self.weights_dim
        self.total_return = 1.0
        return self._get_observation(), {}

    def step(self, action):
        action = np.clip(action, 0, 1)
        action /= action.sum() if action.sum() > 0 else len(action)
        returns = self.returns_df.iloc[self.current_step].values
        reward = np.dot(action, returns)

        window = max(0, self.current_step - 5)
        past_returns = self.returns_df.iloc[window:self.current_step+1].dot(action)
        volatility = max(past_returns.std() if len(past_returns) > 1 else 1e-6, 1e-2)
        reward /= volatility

        transaction_cost = np.sum(np.abs(action - self.prev_action)) * self.cost_rate if self.apply_cost else 0
        reward -= transaction_cost
        self.prev_action = action
        if self.clip_rewards:
            reward = np.clip(reward, -0.03, 0.03)
        log_return = np.log(1 + reward)
        if self.current_step > 0 and self.current_step % self.shock_interval == 0:
            log_return += self.shock_magnitude
        self.total_return *= np.exp(log_return)
        if self.reset_capital and self.current_step > 0 and self.current_step % self.reset_interval == 0:
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
        return np.concatenate([
            self.returns_df.iloc[self.current_step].values,
            self.regime_probs.iloc[self.current_step].values
        ]).astype(np.float32)

# === PPO Variant Training ===
def train_variants():
    returns_df = pd.read_csv("returns_matrix.csv")
    regime_probs = pd.read_csv("hmm_regime_probs.csv")
    weights_dim = returns_df.shape[1]
    seeds = [0, 1, 2, 3, 4]
    variant_configs = {
        "baseline": {"clip_rewards": True, "apply_cost": True, "reset_capital": True},
        "noclip": {"clip_rewards": False, "apply_cost": True, "reset_capital": True},
        "nocost": {"clip_rewards": True, "apply_cost": False, "reset_capital": True},
        "noreset": {"clip_rewards": True, "apply_cost": True, "reset_capital": False}
    }
    for variant, config in variant_configs.items():
        for seed in seeds:
            np.random.seed(seed)
            env = DummyVecEnv([lambda: RegimeAwarePortfolioEnv(
                returns_df, regime_probs, weights_dim,
                clip_rewards=config["clip_rewards"],
                apply_cost=config["apply_cost"],
                reset_capital=config["reset_capital"]
            )])
            model = PPO("MlpPolicy", env, verbose=0, seed=seed, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.90)
            model.learn(total_timesteps=100_000)
            model.save(f"{variant}_seed{seed}")

# === SHAP Explanation ===
def run_shap():
    model = PPO.load("ppo_rl_stable")
    returns_df = pd.read_csv("returns_matrix.csv")
    regime_probs = pd.read_csv("hmm_regime_probs.csv")
    obs_matrix = np.hstack([returns_df.values, regime_probs.values])
    if obs_matrix.shape[0] < 30:
        obs_matrix = np.vstack([obs_matrix] * 4)
    obs_tensor = torch.tensor(obs_matrix, dtype=torch.float32)

    class PolicyWrapper(nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy_net = policy.mlp_extractor.policy_net
            self.final_layer = policy.action_net
        def forward(self, x):
            x = self.policy_net(x)
            return self.final_layer(x)

    wrapped_policy = PolicyWrapper(model.policy)
    background_size = 20
    test_size = 5
    explainer = shap.DeepExplainer(wrapped_policy, obs_tensor[:background_size])
    shap_values = explainer.shap_values(obs_tensor[background_size:background_size + test_size])
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap.summary_plot(
        shap_values,
        obs_matrix[background_size:background_size + test_size],
        feature_names=list(returns_df.columns) + list(regime_probs.columns),
        show=True
    )

# === Run All ===
if __name__ == "__main__":
    print("Training PPO Variants...")
    train_variants()
    print("Generating SHAP Visualizations...")
    run_shap()
