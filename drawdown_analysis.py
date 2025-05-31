import pandas as pd
import numpy as np

df = pd.read_csv('cleaned_real_asset_returns.csv')
assets = ['S&P 500', 'Small Cap', 'Real Estate', 'Gold', 'T-Bill', 'Baa Corporate']
for col in assets:
    df[col] = df[col].str.rstrip('%').astype(float)
df.set_index('Year', inplace=True)

drawdowns = []
for asset in assets:
    cum_ret = (1 + df[asset]/100).cumprod()
    peak = cum_ret.cummax()
    dd = (cum_ret - peak)/peak
    drawdowns.append({'Asset': asset, 'Max Drawdown (%)': round(dd.min()*100,2), 'Year': dd.idxmin()})

drawdown_df = pd.DataFrame(drawdowns)
drawdown_df.to_csv('drawdown_analysis.csv', index=False)
