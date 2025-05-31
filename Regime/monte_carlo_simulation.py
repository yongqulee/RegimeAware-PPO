import pandas as pd
import numpy as np

df = pd.read_csv("regime_clusters_output.csv")
asset_cols = [
    'S&P 500 Return', 'Small Cap Return', 'Real Estate Return',
    'Gold Return', 'T-Bill', 'T-Bill Spread'
]
df[asset_cols] = df[asset_cols] / 100.0
df['Stress_Regime'] = df['Regime_GMM'].apply(lambda x: 1 if x == 0 else 0)

stress = df[df['Stress_Regime'] == 1][asset_cols]
normal = df[df['Stress_Regime'] == 0][asset_cols]
mu_stress, cov_stress = stress.mean().values, stress.cov().values
mu_normal, cov_normal = normal.mean().values, normal.cov().values

P = {0: {0: 0.90, 1: 0.10}, 1: {0: 0.40, 1: 0.60}}
weights = np.array([0.134, 0.019, 0.398, 0.118, 0.0, 0.331])
n_simulations, n_years = 10000, 10
results = []

for _ in range(n_simulations):
    regime = 0
    cum_ret = 1
    for _ in range(n_years):
        r = np.random.multivariate_normal(
            mu_normal if regime == 0 else mu_stress,
            cov_normal if regime == 0 else cov_stress
        )
        cum_ret *= (1 + np.dot(weights, r))
        regime = np.random.choice([0, 1], p=[P[regime][0], P[regime][1]])
    results.append((cum_ret - 1) * 100)

results = np.array(results)
print("Mean Return: {:.2f}%".format(results.mean()))
print("95% CI: [{:.2f}%, {:.2f}%]".format(np.percentile(results, 2.5), np.percentile(results, 97.5)))
