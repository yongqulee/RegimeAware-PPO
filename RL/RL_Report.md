Reinforcement Learning Portfolio Optimization (RL/)

This section documents the implementation and evaluation of a regime-aware PPO-based reinforcement learning agent for dynamic asset allocation using historical return data and latent market regimes.

Contents

ppo_rl_pipeline.py: Initial PPO agent using simple reward structure

ppo_rl_stable.py: Final PPO implementation with risk-adjusted rewards, transaction costs, clipping, capital resets, and shock injection

ppo_rl_stable.zip: Trained model weights

returns_matrix.csv: Damodaran annual asset class returns

hmm_regime_probs.csv: Hidden Markov Model (HMM) derived regime probabilities

Plots:

ppo_vs_equal_portfolio_growth.png: PPO vs Equal-Weight value evolution

rolling_cagr_comparison_all_strategies.png: 10y/20y/30y CAGR across strategies

rolling_cagr_stress_overlay.png: PPO CAGR with stress year overlays

PPO Agent Overview

Environment: Custom Gym RegimeAwarePortfolioEnv

Observations: Annual asset returns + HMM regime probability vector (dimensionality: n_assets + n_regimes)

Action Space: Continuous allocation weights for each asset (softmax normalized)

Reward Function:

Rolling volatility-normalized return

Transaction cost penalty (cost_rate = 0.001)

Clipped to ±3% before log return transformation

Capital reset every 30 steps (to avoid exponential bias)

Stress shock of -5% injected every 25 steps

Final Evaluation Results

Metric

PPO (Stable)

Equal-Weight

Sharpe-Optimized

Sharpe Ratio

1.0677

0.4152

0.5106

Sortino Ratio

1.1970

0.7771

0.7105

Max Drawdown

-72.58%

-28.91%

-24.55%

Final Portfolio Value

$1.113 × 10¹²

$43.04

$69.11

The PPO agent outperforms both baselines in terms of Sharpe and Sortino ratios. The max drawdown is higher, reflecting increased exposure to high-return regimes, which also increases risk.

Rolling CAGR Analysis

Using rolling windows of 10, 20, and 30 years, we observe:

PPO 10y CAGR ranges from 28% to 49%

PPO 20y CAGR remains above 30% until final third, then drops near 20%

Equal and Optimized CAGR remain between 3% to 7% throughout

This confirms PPO’s superior ability to exploit regime dynamics over long-term horizons.

See figure: rolling_cagr_comparison_all_strategies.png

Implementation Notes

PPO trained using stable-baselines3 with gamma = 0.90

Used a custom vectorized environment with horizon = 30

Evaluated on full length of historical return series from Damodaran (annual)

Regimes derived from Gaussian HMM trained separately (3-state model)

Recommendations

Use ppo_rl_stable.py for deployment or further tuning

Adjust shock_interval, reset_interval, and cost_rate for scenario testing

Extend to out-of-sample or forward-simulated stress testing for robustness

