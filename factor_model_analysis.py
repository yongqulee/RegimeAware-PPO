import pandas as pd
import statsmodels.api as sm

returns = pd.read_csv('cleaned_real_asset_returns.csv')
ff_factors = pd.read_csv('FamaFrench_factors.csv')

returns['Optimized_Portfolio'] = returns[['S&P 500', 'Small Cap', 'Real Estate', 'Gold', 'T-Bill', 'Baa Corporate']].mean(axis=1)
returns.set_index('Year', inplace=True)
ff_factors.set_index('Date', inplace=True)

returns.index = pd.to_datetime(returns.index, format='%Y') + pd.offsets.YearBegin()
data = ff_factors.join(returns['Optimized_Portfolio'], how='inner')
data['Excess_Return'] = data['Optimized_Portfolio'] - data['RF']

X = data[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']]
y = data['Excess_Return']
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
with open('factor_model_summary.txt', 'w') as f:
    f.write(model.summary().as_text())
