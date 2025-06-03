# === A2C (No Regime) Portfolio Trainer ===

import pandas as pd
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from gym import Env, spaces
import matplotlib.pyplot as plt

# === Load Returns ===
returns_df = pd.read_csv("returns_matrix.csv")
returns = returns_df.values
n_assets = returns.shape[1]

# === Environment ===
class A2CEnv(Env):
    def __init__(self, returns):
        super().__init__()
        self.returns = returns
        self.n_assets = returns.shape[1]
        self.current_step = 1
        self.nav = 1.0
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_assets,), dtype=np.float32)

    def reset(self):
        self.current_step = 1
        self.nav = 1.0
        return self.returns[self.current_step - 1]

    def step(self, action):
        action = np.nan_to_num(action)
        action = action / (np.sum(action) + 1e-8)
        reward = np.dot(action, self.returns[self.current_step])
        self.nav *= (1 + reward)
        self.current_step += 1
        done = self.current_step >= len(self.returns) - 1
        return self.returns[self.current_step - 1], reward, done, {}

# === Setup and Train ===
env = DummyVecEnv([lambda: A2CEnv(returns)])
model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
model.save("a2c_no_regime")

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
plt.title("A2C (No Regime) NAV Curve")
plt.xlabel("Step")
plt.ylabel("NAV")
plt.grid(True)
plt.savefig("a2c_nav_curve.png")
plt.show()

# === Metrics ===
returns = np.diff(navs) / navs[:-1]
sharpe = np.mean(returns) / (np.std(returns) + 1e-8)
peak = np.maximum.accumulate(navs)
drawdown = (peak - navs) / (peak + 1e-8)
max_dd = np.max(drawdown)

print(f"Sharpe Ratio: {sharpe:.4f}")
print(f"Max Drawdown: {max_dd:.4f}")
