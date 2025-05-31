import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture

regime_df = pd.read_csv("regime_clusters_output.csv")
asset_cols = [
    'S&P 500 Return', 'Small Cap Return', 'Real Estate Return',
    'Gold Return', 'T-Bill', 'T-Bill Spread'
]
regime_df[asset_cols] = regime_df[asset_cols] / 100.0
regime_df['RiskPremium'] = regime_df['S&P 500 Return'] - regime_df['T-Bill']
regime_df['InflationProxy'] = regime_df['T-Bill'] - regime_df['T-Bill Spread']

gmm = GaussianMixture(n_components=2, random_state=42)
macro_features = regime_df[['RiskPremium', 'InflationProxy']].dropna()
gmm_labels = gmm.fit_predict(macro_features)
regime_df = regime_df.loc[macro_features.index].copy()
regime_df['Stress_Regime'] = gmm_labels

stress = regime_df[regime_df['Stress_Regime'] == 1][asset_cols].clip(-1, 1)
normal = regime_df[regime_df['Stress_Regime'] == 0][asset_cols].clip(-1, 1)
mu_stress, cov_stress = stress.mean().values, stress.cov().values
mu_normal, cov_normal = normal.mean().values, normal.cov().values

P = pd.crosstab(regime_df['Stress_Regime'].shift(1), regime_df['Stress_Regime']).div(
    regime_df['Stress_Regime'].shift(1).value_counts(), axis=0).fillna(0)
P = {i: P.loc[i].to_dict() for i in P.index}

weights = np.array([0.134, 0.019, 0.398, 0.118, 0.0, 0.331])
n_sim, n_years, results = 10000, 20, []

for _ in range(n_sim):
    regime = np.random.choice([0, 1])
    total_log_return = 0
    for _ in range(n_years):
        r = np.random.multivariate_normal(mu_normal if regime == 0 else mu_stress,
                                          cov_normal if regime == 0 else cov_stress)
        total_log_return += np.log1p(np.dot(weights, r))
        regime = np.random.choice([0, 1], p=[P[regime][0], P[regime][1]])
    results.append(np.expm1(total_log_return) * 100)

results = np.array(results)
print(f"Mean: {results.mean():.2f}% | Median: {np.median(results):.2f}%")
print(f"VaR (5%): {np.percentile(results, 5):.2f}%")
print(f"CVaR: {results[results <= np.percentile(results, 5)].mean():.2f}%")
