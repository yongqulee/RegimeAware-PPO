import numpy as np
import pandas as pd
from scipy.optimize import minimize

df = pd.read_csv('cleaned_real_asset_returns.csv')
assets = ['S&P 500', 'Small Cap', 'Real Estate', 'Gold', 'T-Bill', 'Baa Corporate']
for col in assets:
    df[col] = df[col].str.rstrip('%').astype(float)

returns = df[assets].values / 100
mean_returns = returns.mean(axis=0)
downside_returns = np.minimum(returns, 0)
downside_cov = np.cov(downside_returns.T)

def negative_sortino(weights, mean_returns, downside_cov):
    ret = weights @ mean_returns
    downside_std = np.sqrt(weights @ downside_cov @ weights)
    return -ret / downside_std if downside_std else np.inf

cons = {'type':'eq', 'fun':lambda x: sum(x)-1}
bounds = [(0,1)]*len(assets)
initial = np.array(len(assets)*[1/len(assets)])

res = minimize(negative_sortino, initial, args=(mean_returns, downside_cov), method='SLSQP', bounds=bounds, constraints=cons)

weights_df = pd.DataFrame({'Asset': assets, 'Sortino Weight': res.x})
weights_df.to_csv('sortino_portfolio_weights.csv', index=False)
