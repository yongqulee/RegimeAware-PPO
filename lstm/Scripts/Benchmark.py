# === Benchmark Portfolio Evaluation ===

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Load Returns ===
returns_df = pd.read_csv("returns_matrix.csv")
avg_returns = returns_df.mean(axis=1).values

# === NAV Calculation ===
navs = [1.0]
for r in avg_returns:
    navs.append(navs[-1] * (1 + r))

# === Plot ===
plt.plot(navs)
plt.title("Benchmark Portfolio NAV Curve")
plt.xlabel("Step")
plt.ylabel("NAV")
plt.grid(True)
plt.savefig("benchmark_nav_curve.png")
plt.show()

# === Metrics ===
returns = np.diff(navs) / navs[:-1]
sharpe = np.mean(returns) / (np.std(returns) + 1e-8)
peak = np.maximum.accumulate(navs)
drawdown = (peak - navs) / (peak + 1e-8)
max_dd = np.max(drawdown)

print(f"Benchmark - Sharpe Ratio: {sharpe:.4f}")
print(f"Benchmark - Max Drawdown: {max_dd:.4f}")
