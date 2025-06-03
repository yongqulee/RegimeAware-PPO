# === Recurrent PPO (LSTM) - Final Portfolio Trainer ===

import pandas as pd
import numpy as np
from sb3_contrib.ppo_recurrent import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from gym import Env, spaces
import matplotlib.pyplot as plt

# === Load Data ===
returns_df = pd.read_csv("returns_matrix.csv")
trans_regimes = pd.read_csv("transformer_regime_predictions.csv").values.argmax(axis=1)
returns = returns_df.values
n_assets = returns.shape[1]

# === Environment with Regime ===
class RecurrentEnv(Env):
    def __init__(self, returns, regimes):
        super().__init__()
        self.returns = returns
        self.regimes = regimes
        self.n_assets = returns.shape[1]
        self.current_step = 1
        self.nav = 1.0
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_assets + 1,), dtype=np.float32)

    def reset(self):
        self.current_step = 1
        self.nav = 1.0
        return self._get_obs()

    def _get_obs(self):
        obs = list(self.returns[self.current_step - 1])
        obs.append(self.regimes[self.current_step])
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        action = np.nan_to_num(action)
        action = action / (np.sum(action) + 1e-8)
        reward = np.dot(action, self.returns[self.current_step])
        self.nav *= (1 + reward)
        self.current_step += 1
        done = self.current_step >= len(self.returns) - 1
        return self._get_obs(), reward, done, {}

# === Setup and Train ===
env = DummyVecEnv([lambda: RecurrentEnv(returns, trans_regimes)])
model = RecurrentPPO(
    policy=MlpLstmPolicy,
    env=env,
    verbose=1,
    learning_rate=1e-4,
    n_steps=512,
    batch_size=64,
    n_epochs=10
)
model.learn(total_timesteps=250_000)
model.save("recurrent_ppo_final")

# === Evaluate ===
obs = env.reset()
done = False
navs = [1.0]

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    navs.append(env.get_attr("nav")[0])

# === Plot NAV ===
plt.plot(navs)
plt.title("Recurrent PPO Final NAV Curve")
plt.xlabel("Step")
plt.ylabel("NAV")
plt.grid(True)
plt.savefig("recurrent_ppo_nav_curve.png")
plt.show()

# === Metrics ===
returns = np.diff(navs) / navs[:-1]
sharpe = np.mean(returns) / (np.std(returns) + 1e-8)
peak = np.maximum.accumulate(navs)
drawdown = (peak - navs) / (peak + 1e-8)
max_dd = np.max(drawdown)

print(f"Sharpe Ratio: {sharpe:.4f}")
print(f"Max Drawdown: {max_dd:.4f}")
