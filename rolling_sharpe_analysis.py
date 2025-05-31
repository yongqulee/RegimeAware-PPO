import pandas as pd
import numpy as np

df = pd.read_csv('cleaned_real_asset_returns.csv')

percent_cols = ['S&P 500', 'Small Cap', 'T-Bill', '10Y Treasury', 'Baa Corporate', 'Real Estate', 'Gold']
for col in percent_cols:
    df[col] = df[col].str.rstrip('%').astype(float)

df.set_index('Year', inplace=True)

windows = {'2y': 2, '5y': 5, '10y': 10}
assets = ['S&P 500', 'Small Cap', '10Y Treasury', 'Baa Corporate', 'Real Estate', 'Gold']
rf_col = 'T-Bill'

results = []
for label, win in windows.items():
    for asset in assets:
        excess_returns = df[asset] - df[rf_col]
        rolling_mean = excess_returns.rolling(window=win).mean()
        rolling_std = excess_returns.rolling(window=win).std(ddof=0)
        rolling_sharpe = rolling_mean / rolling_std
        rolling_sharpe = rolling_sharpe.dropna()
        for year, value in rolling_sharpe.items():
            results.append({'Year': year, 'Asset': asset, 'Window': label, 'Rolling Sharpe': round(value, 4)})

rolling_sharpe_df = pd.DataFrame(results)
rolling_sharpe_df.to_csv('rolling_sharpe_output.csv', index=False)
