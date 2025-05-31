# Regime-Aware Portfolio Simulation – Results Summary

This document summarizes the results from multiple simulations and analyses using regime-aware modeling techniques, including KMeans, GMM, HMM, and Monte Carlo simulations under both fixed and macro-informed transitions.

---

## Regime Detection Summary

- Models Used: KMeans, GMM, HMM
- Number of Regimes: 3
- Clustering Input: Volatility, drawdown, spreads, return-based engineered features from `regime_features.csv`
- Stress Year Detection:

| Year | Regime_KMeans | Regime_GMM | Regime_HMM |
|------|---------------|------------|------------|
| 1974 | 0             | 0          | 0          |
| 1987 | 1             | 1          | 2          |
| 2001 | 1             | 1          | 2          |
| 2008 | 0             | 0          | 0          |
| 2020 | 1             | 1          | 2          |

Interpretation:
- GMM Regime 0 = systemic crashes (1974, 2008)
- HMM Regime 2 = quick shocks (1987, 2001, 2020)

---

## Monte Carlo Simulations

Assumptions:
- 6-asset portfolio
- Stress regime derived from GMM
- Regime switching matrix:
  - Normal: 90% stay, 10% go to stress
  - Stress: 60% stay, 40% recover
- Log-return compounding

Output Files:
- `regime_monte_carlo_Optimized_10y.csv`
- `regime_monte_carlo_EqualWeight_10y.csv`
- and corresponding 20y and 30y variants

---

## Simulation Results Summary

| Portfolio     | Horizon | Mean   | Median | 95% CI                         | VaR (5%) | CVaR (5%) |
|---------------|---------|--------|--------|--------------------------------|----------|-----------|
| Optimized     | 10y     | 25.15% | 24.68% | [-9.53%, 63.09%]               | -4.29%   | -11.31%   |
| Equal Weight  | 10y     | 69.64% | 64.76% | [-8.11%, 173.03%]              | 1.67%    | -10.09%   |
| Optimized     | 20y     | 54.61% | 52.69% | [-3.10%, 121.53%]              | 5.72%    | -5.08%    |
| Equal Weight  | 20y     | 175.59%| 156.29%| [12.87%, 442.49%]              | 30.35%   | 9.24%     |
| Optimized     | 30y     | 91.85% | 86.55% | [8.76%, 205.30%]               | 20.28%   | 6.40%     |
| Equal Weight  | 30y     | 358.90%| 311.48%| [56.88%, 923.69%]              | 84.20%   | 49.40%    |

Interpretation:
- Equal-weight portfolios outperform in long-term growth.
- Optimized portfolios are more defensive in shorter time frames.
- Regime-switching logic captures realistic drawdowns.

---

## Enhanced Monte Carlo (GMM on Macro)

- Regime drivers: Risk premium and yield spread
- Output file: `enhanced_monte_carlo_results.csv`

| Metric         | Value       |
|----------------|-------------|
| Mean Return    | 45.71%      |
| Median Return  | 42.68%      |
| 95% CI         | [-7.63%, 113.79%] |
| VaR (5%)       | -0.65%      |
| CVaR (5%)      | -9.22%      |

Interpretation:
Adds macro-informed realism to regime simulation and reflects quick recoveries post-crisis.

---

## Reinforcement Learning Environment (Next Stage)

- PPO agent trains on a custom Gym environment with:
  - Return vector
  - HMM regime probabilities
- Observation = [returns_t, HMM_state_probs]
- Action = portfolio weights

Trained model: `ppo_regime_portfolio.zip`

---

## Conclusion

- Regime detection aligns well with financial history.
- Monte Carlo outputs reflect realistic risk-reward trade-offs.
- Enhanced simulation adds macro-awareness.
- Reinforcement Learning training environment is ready for deployment and benchmarking.

