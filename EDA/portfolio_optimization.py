import pandas as pd
import numpy as np
from scipy.optimize import minimize

df = pd.read_csv('cleaned_real_asset_returns.csv')
assets = ['S&P 500', 'Small Cap', 'Real Estate', 'Gold', 'T-Bill', 'Baa Corporate']
for col in assets:
    df[col] = df[col].str.rstrip('%').astype(float)

returns = df[assets].values / 100
mean_returns = returns.mean(axis=0)
cov_matrix = np.cov(returns.T)

def negative_sharpe(weights, mean_returns, cov_matrix, rf=0.01):
    ret = weights @ mean_returns
    std = np.sqrt(weights @ cov_matrix @ weights)
    return -(ret - rf) / std

cons = {'type': 'eq', 'fun': lambda x: np.sum(x)-1}
bounds = [(0,1)] * len(assets)
init = np.array(len(assets)*[1/len(assets)])

result = minimize(negative_sharpe, init, args=(mean_returns, cov_matrix), method='SLSQP', bounds=bounds, constraints=cons)

weights_df = pd.DataFrame({'Asset': assets, 'Weight': result.x})
weights_df.to_csv('optimized_portfolio_weights.csv', index=False)
