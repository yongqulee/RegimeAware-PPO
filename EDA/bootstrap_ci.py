import pandas as pd
import numpy as np

df = pd.read_csv('cleaned_real_asset_returns.csv')
for col in df.columns[1:]:
    df[col] = df[col].str.rstrip('%').astype(float)

df.set_index('Year', inplace=True)
returns = df / 100

def bootstrap_ci(data, n=1000, ci=95):
    boot = [(1 + data.sample(frac=1, replace=True)).prod() - 1 for _ in range(n)]
    lower, upper = np.percentile(boot, [(100-ci)/2, 100-(100-ci)/2])
    return lower, upper

ci_records = []
for col in returns.columns:
    low, high = bootstrap_ci(returns[col])
    ci_records.append({'Asset': col, 'CI Lower': low, 'CI Upper': high})

ci_df = pd.DataFrame(ci_records)
ci_df.to_csv('bootstrap_confidence_intervals.csv', index=False)
