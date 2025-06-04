import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C
from sb3_contrib.ppo_recurrent import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym import Env

from scripts.recurrent_ppo_lstm import FullHybridEnv, returns_df_extended, weights_dim, benchmark_returns

MODEL_PATHS = {
    "A2C_No_Regime": "models/a2c_no_regime.zip",
    "PPO_GMM": "models/ppo_gmm.zip",
    "PPO_Transformer": "models/ppo_transformer.zip",
    "RecurrentPPO_LSTM_Final": "models/ppo_lstm_full_hybrid.zip",
}

def make_eval_env():
    return DummyVecEnv([lambda: FullHybridEnv(returns_df_extended, weights_dim)])

def evaluate_model(model, env, label="Model"):
    obs = env.reset()
    portfolio_returns = []
    nav = [1.0]

    for _ in range(180):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        portfolio_returns.append(reward[0])
        nav.append(nav[-1] + reward[0])
        if done[0]:
            obs = env.reset()

    returns = np.array(portfolio_returns)
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8)
    cumulative = np.cumsum(returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    max_drawdown = np.max(drawdown)

    plt.plot(nav, label=label)
    return sharpe, max_drawdown, returns

env = make_eval_env()
results = {}

for name, path in MODEL_PATHS.items():
    if "Recurrent" in name:
        model = RecurrentPPO.load(path, env=env)
    elif "A2C" in name:
        model = A2C.load(path, env=env)
    else:
        model = PPO.load(path, env=env)
    
    sharpe, max_dd, _ = evaluate_model(model, env, label=name)
    results[name] = (sharpe, max_dd)

benchmark_returns_truncated = benchmark_returns[:180]
benchmark_sharpe = np.mean(benchmark_returns_truncated) / (np.std(benchmark_returns_truncated) + 1e-8)
cumulative = np.cumsum(benchmark_returns_truncated)
peak = np.maximum.accumulate(cumulative)
drawdown = peak - cumulative
benchmark_dd = np.max(drawdown)
plt.plot(np.cumsum(benchmark_returns_truncated) + 1, label="Benchmark")

plt.title("Cumulative NAV: RL Models vs Benchmark")
plt.xlabel("Timestep")
plt.ylabel("NAV")
plt.grid(True)
plt.legend()
plt.savefig("plots/model_comparison_nav.png")
plt.show()

print("\n=== Evaluation Results ===")
print("| Model                      | Sharpe Ratio | Max Drawdown |")
print("|---------------------------|--------------|---------------|")
for model, (sharpe, dd) in results.items():
    print(f"| {model:<26} | {sharpe:>12.3f} | {dd:>13.3f} |")
print(f"| {'Benchmark':<26} | {benchmark_sharpe:>12.3f} | {benchmark_dd:>13.3f} |")
