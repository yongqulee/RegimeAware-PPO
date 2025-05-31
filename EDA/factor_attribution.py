import pandas as pd
import statsmodels.api as sm

returns = pd.read_csv('cleaned_real_asset_returns.csv')
ff_factors = pd.read_csv('FamaFrench_factors.csv')

returns['Portfolio'] = returns[['S&P 500', 'Small Cap', 'Real Estate', 'Gold', 'T-Bill', 'Baa Corporate']].mean(axis=1)
returns.set_index('Year', inplace=True)
returns.index = pd.to_datetime(returns.index, format='%Y') + pd.offsets.YearBegin()

ff_factors.set_index('Date', inplace=True)

data = ff_factors.join(returns['Portfolio'], how='inner')
data['Excess_Return'] = data['Portfolio'] - data['RF']

X = sm.add_constant(data[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']])
y = data['Excess_Return']

model = sm.OLS(y, X).fit()

with open('factor_attribution_summary.txt', 'w') as f:
    f.write(model.summary().as_text())
