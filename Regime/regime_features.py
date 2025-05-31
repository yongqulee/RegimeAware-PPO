import pandas as pd
import numpy as np

df = pd.read_csv("cleaned_real_asset_returns.csv")
percent_cols = df.columns.drop('Year')
df[percent_cols] = df[percent_cols].replace('%', '', regex=True).astype(float)
df.set_index('Year', inplace=True)

features = pd.DataFrame(index=df.index)
features['S&P 500 Return'] = df['S&P 500']
features['Small Cap Return'] = df['Small Cap']
features['Real Estate Return'] = df['Real Estate']
features['Gold Return'] = df['Gold']
features['T-Bill'] = df['T-Bill']
features['T-Bill Spread'] = df['T-Bill'] - df['10Y Treasury']

for win in [2, 5]:
    features[f'Vol_S&P_500_{win}y'] = df['S&P 500'].rolling(window=win).std()
    features[f'Vol_Small_Cap_{win}y'] = df['Small Cap'].rolling(window=win).std()

cum_return = (1 + df[['S&P 500', 'Small Cap']] / 100).cumprod()
peak = cum_return.cummax()
drawdown = (cum_return - peak) / peak
features['Drawdown_S&P_500'] = drawdown['S&P 500'] * 100
features['Drawdown_Small_Cap'] = drawdown['Small Cap'] * 100
features.dropna(inplace=True)

features.to_csv("regime_features.csv")
