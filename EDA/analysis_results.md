
# Analysis Results

## Sharpe Ratio Analysis

| Asset | Sharpe (Stress) | Sharpe (Non-Stress) |
|-------|-----------------|---------------------|
| S&P 500 | -0.8399 | 0.5669 |
| Small Cap | -0.4219 | 0.4230 |
| 10Y Treasury | 0.1781 | 0.2008 |
| Baa Corporate | -0.3945 | 0.5449 |
| Real Estate | -0.1551 | 0.1760 |
| Gold | 0.5199 | 0.1374 |

## PCA Resilience Scores

| Asset | PCA Resilience Score |
|-------|----------------------|
| Real Estate | 0.982 |
| T-Bill | 0.974 |
| Gold | 0.650 |
| S&P 500 | 0.602 |
| Small Cap | 0.000 |

## Portfolio Optimization (Sharpe)

| Asset | Optimized Weight |
|-------|------------------|
| S&P 500 | 0.134 |
| Small Cap | 0.019 |
| Real Estate | 0.398 |
| Gold | 0.118 |
| T-Bill | 0.000 |
| Baa Corporate | 0.331 |

## Portfolio Optimization (Sortino)

| Asset | Sortino Optimized Weight |
|-------|--------------------------|
| S&P 500 | 0.000 |
| Small Cap | 0.137 |
| Real Estate | 0.235 |
| Gold | 0.153 |
| T-Bill | 0.000 |
| Baa Corporate | 0.474 |

## Drawdown Analysis

| Asset | Max Drawdown (%) | Year |
|-------|------------------|------|
| Gold | -77.10 | 2001 |
| Small Cap | -82.71 | 1931 |
| T-Bill | -45.78 | 1951 |
| Baa Corporate | -30.10 | 1981 |
| S&P 500 | -54.84 | 1931 |
| Real Estate | -34.56 | 2011 |

## Factor Attribution Regression

- **R-squared:** 0.040
- **Significant Factors:**
  - Market-RF (Mkt-RF): p-value = 0.000, coef = 0.207
- **Insignificant Factors:** SMB, HML, RMW, CMA, Momentum

## Rolling Factor Betas (Summary)

- Rolling betas computed over a 60-month window show variability in factor exposure.

## Residual Diagnostics

- **Ljung-Box Test:** Significant autocorrelation detected.
- **Breusch-Pagan Test:** LM Statistic: significant, indicating heteroskedasticity.

## Bootstrap Confidence Intervals

| Asset | 95% CI Lower | 95% CI Upper |
|-------|--------------|--------------|
| Optimized Portfolio | 463.04% | 5059.73% |
| Equal Weighted Portfolio | 1198.24% | 48203.96% |
| Sortino Optimized | 1027.00% | 24966.54% |
