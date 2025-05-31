import numpy as np
import pandas as pd

df = pd.read_csv('cleaned_real_asset_returns.csv')
percent_cols = ['S&P 500', 'Small Cap', 'T-Bill', '10Y Treasury', 'Baa Corporate', 'Real Estate', 'Gold']
df[percent_cols] = df[percent_cols].replace('%', '', regex=True).astype(float)
df.set_index('Year', inplace=True)

assets = ['S&P 500', 'Small Cap', 'Real Estate', 'Gold', 'T-Bill', 'Baa Corporate']
returns = df[assets] / 100

def bootstrap_cumret(data, n_iter=1000, ci=95):
    boot = [(1 + data.sample(frac=1, replace=True)).prod() - 1 for _ in range(n_iter)]
    lower = np.percentile(boot, (100 - ci) / 2)
    upper = np.percentile(boot, 100 - (100 - ci) / 2)
    return lower, upper

results = []
for asset in assets:
    lower, upper = bootstrap_cumret(returns[asset])
    results.append({'Asset': asset, '95% CI Lower': lower, '95% CI Upper': upper})

bootstrap_df = pd.DataFrame(results)
bootstrap_df.to_csv('monte_carlo_bootstrap.csv', index=False)
