import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('cleaned_real_asset_returns.csv')
percent_cols = ['S&P 500', 'Small Cap', 'T-Bill', '10Y Treasury', 'Baa Corporate', 'Real Estate', 'Gold']
for col in percent_cols:
    df[col] = df[col].str.rstrip('%').astype(float)
df.set_index('Year', inplace=True)

stress_years = [1931, 1974, 1987, 2001, 2008, 2020]
assets = ['S&P 500', 'Small Cap', 'Real Estate', 'Gold', 'T-Bill', 'Baa Corporate']
records = []
for year in stress_years:
    window = range(year - 2, year + 3)
    if all(y in df.index for y in window):
        for asset in assets:
            vol = df.loc[window, asset].std()
            cum_ret = (1 + df.loc[window, asset]/100).prod() - 1
            cagr = (1 + df.loc[year+1:year+5, asset]/100).prod()**(1/5) - 1
            records.append({'Asset': asset, 'Year': year, 'Volatility': vol, 'Cumulative Return': cum_ret, 'CAGR': cagr})

resilience_df = pd.DataFrame(records)
resilience_df['Vol_inv'] = -resilience_df['Volatility']
scaler = MinMaxScaler()
resilience_df['Score'] = scaler.fit_transform(resilience_df[['Vol_inv', 'Cumulative Return', 'CAGR']]).mean(axis=1)
resilience_df.to_csv('asset_resilience_scores.csv', index=False)
