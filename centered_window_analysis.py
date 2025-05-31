import pandas as pd
import numpy as np

df = pd.read_csv('cleaned_real_asset_returns.csv')

percent_cols = ['S&P 500', 'Small Cap', 'T-Bill', '10Y Treasury', 'Baa Corporate', 'Real Estate', 'Gold']
for col in percent_cols:
    df[col] = df[col].str.rstrip('%').astype(float)

df.set_index('Year', inplace=True)

stress_years = [1931, 1974, 1987, 2001, 2008, 2020]
windows = {'2y': 1, '5y': 2, '10y': 5}

assets = ['S&P 500', 'Small Cap', 'Real Estate', 'Gold', '10Y Treasury', 'Baa Corporate']
rf_col = 'T-Bill'

records = []
for label, k in windows.items():
    for year in stress_years:
        years = list(range(year - k, year + k + 1))
        if all(y in df.index for y in years):
            for asset in assets:
                excess_returns = df.loc[years, asset] - df.loc[years, rf_col]
                mean = excess_returns.mean()
                std = excess_returns.std(ddof=0)
                sharpe = mean / std if std != 0 else np.nan
                records.append({'Asset': asset, 'Window': label, 'Year': year, 'Centered Sharpe': round(sharpe, 4)})

sharpe_df = pd.DataFrame(records)
sharpe_df.to_csv('centered_window_sharpe.csv', index=False)
