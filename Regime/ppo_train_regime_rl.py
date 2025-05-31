import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from regime_rl_env import RegimeAwarePortfolioEnv
from stable_baselines3.common.vec_env import DummyVecEnv

returns_df = pd.read_csv("returns_matrix.csv")
regime_probs = pd.read_csv("hmm_regime_probs.csv")
weights_dim = returns_df.shape[1]
horizon = 20

env = DummyVecEnv([lambda: RegimeAwarePortfolioEnv(returns_df, regime_probs, weights_dim, horizon)])
check_env(RegimeAwarePortfolioEnv(returns_df, regime_probs, weights_dim, horizon), warn=True)

model = PPO("MlpPolicy", env, verbose=1, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99)
model.learn(total_timesteps=100_000)
model.save("ppo_regime_portfolio")
