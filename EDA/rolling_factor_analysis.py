import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan

returns = pd.read_csv('cleaned_real_asset_returns.csv')
ff_factors = pd.read_csv('FamaFrench_factors.csv')

returns['Portfolio'] = returns[['S&P 500', 'Small Cap', 'Real Estate', 'Gold', 'T-Bill', 'Baa Corporate']].mean(axis=1)
returns.set_index('Year', inplace=True)
returns.index = pd.to_datetime(returns.index, format='%Y') + pd.offsets.YearBegin()
ff_factors.set_index('Date', inplace=True)

data = ff_factors.join(returns['Portfolio'], how='inner').dropna()
data['Excess_Return'] = data['Portfolio'] - data['RF']

window = 60
betas = []
for i in range(window, len(data)):
    y_roll = data['Excess_Return'].iloc[i-window:i]
    X_roll = sm.add_constant(data[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']].iloc[i-window:i])
    model = sm.OLS(y_roll, X_roll).fit()
    betas.append(model.params.values)

betas_df = pd.DataFrame(betas, columns=['Intercept','Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom'], index=data.index[window:])
betas_df.to_csv('rolling_factor_betas.csv')

model_full = sm.OLS(data['Excess_Return'], sm.add_constant(data[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']])).fit()
residuals = model_full.resid
lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
bp_test = het_breuschpagan(residuals, model_full.model.exog)

with open('factor_diagnostics.txt','w') as f:
    f.write(lb_test.to_string())
    f.write('\n\nBreusch-Pagan LM Statistic: {:.3f}, p-value: {:.3f}'.format(bp_test[0], bp_test[1]))
