import pandas as pd
import numpy as np

df = pd.read_csv('cleaned_real_asset_returns.csv')

percent_cols = ['S&P 500', 'Small Cap', 'T-Bill', '10Y Treasury', 'Baa Corporate', 'Real Estate', 'Gold']
for col in percent_cols:
    df[col] = df[col].str.rstrip('%').astype(float)

df.set_index('Year', inplace=True)

stress_years = [1931, 1974, 1987, 2001, 2008, 2020]

df_stress = df.loc[stress_years]
df_nonstress = df.drop(stress_years)

def compute_sharpe(data, asset, rf):
    excess = data[asset] - data[rf]
    return excess.mean() / excess.std(ddof=0)

assets = ['S&P 500', 'Small Cap', '10Y Treasury', 'Baa Corporate', 'Real Estate', 'Gold']
rf_column = 'T-Bill'

results = []
for asset in assets:
    stress_sharpe = compute_sharpe(df_stress, asset, rf_column)
    nonstress_sharpe = compute_sharpe(df_nonstress, asset, rf_column)
    results.append({'Asset': asset, 'Sharpe (Stress)': round(stress_sharpe, 4), 'Sharpe (Non-Stress)': round(nonstress_sharpe, 4)})

sharpe_df = pd.DataFrame(results)
sharpe_df.to_csv('sharpe_ratios_stress.csv', index=False)
