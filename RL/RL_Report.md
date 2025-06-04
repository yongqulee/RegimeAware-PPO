# Reinforcement Learning Portfolio Optimization (`RL/`)

This section documents the implementation and evaluation of a **regime-aware PPO-based reinforcement learning agent** for dynamic asset allocation using historical return data and latent market regimes.

---

## Contents

- `ppo_rl_pipeline.py`: Initial PPO agent using simple reward structure
- `ppo_rl_stable.py`: Final PPO implementation with:
  - volatility-adjusted rewards
  - transaction costs
  - reward clipping
  - capital reset
  - market shocks
- `ppo_full_pipeline.py`: Full training and SHAP analysis pipeline
- `returns_matrix.csv`: Damodaran asset class returns
- `hmm_regime_probs.csv`: Regime probabilities (HMM-derived)
- `stable/`, `nocost/`, `noreset/`: Trained weights for each PPO variant (5 seeds each)
- 📊 Evaluation Figures:
  - `ppo_vs_equal_portfolio_growth.png`
  - `rolling_cagr_comparison_all_strategies.png`
  - `rolling_cagr_stress_overlay.png`
  - `KDECAGR.png` (KDE of CAGR for Equal, Momentum, Sharpe-opt)
  - `SHAP.png` (SHAP summary plot)

---

## PPO Agent Overview

- **Environment**: `RegimeAwarePortfolioEnv` (custom Gym)
- **Observations**: Returns + regime probabilities (dim = assets + HMM regimes)
- **Action Space**: Continuous asset weights (Box)
- **Reward Function**:
  - Sharpe-style return-to-volatility ratio
  - Penalty for allocation change (transaction cost)
  - Reward clipping (±3%)
  - Capital reset every 30 steps
  - -5% market shock every 25 steps

---

## Final Evaluation Results

| Metric                | PPO (Stable) | Equal-Weight | Sharpe-Opt |
|----------------------|--------------|--------------|------------|
| Sharpe Ratio          | 1.0677       | 0.4152       | 0.5106     |
| Sortino Ratio         | 1.1970       | 0.7771       | 0.7105     |
| Max Drawdown          | -72.58%      | -28.91%      | -24.55%    |
| Final Value (log)     | \$1.113×10¹² | \$43.04      | \$69.11    |

---

## Rolling CAGR Results

- PPO shows **10y CAGR**: 28% to 49%
- PPO **20y CAGR** remains above 30% for most years
- Equal and Optimized CAGR: 3% to 7% throughout
- See `rolling_cagr_comparison_all_strategies.png` and `rolling_cagr_stress_overlay.png`

---

## Ablation Study Results

PPO variants across 5 seeds:

| Variant   | Sharpe | Sortino   | Max DD   | CAGR      |
|-----------|--------|-----------|----------|-----------|
| Baseline  | 27.57  | 412737.20 | -0.0309  | 625.73    |
| Noclip    | ~5.03  | 1754.51   | -0.1489  | 2.17e+89  |
| Nocost    | 27.57  | 412737.20 | -0.0309  | 625.73    |
| Noreset   | 27.57  | 412737.20 | -0.0309  | 625.73    |

Statistical test results (Wilcoxon vs Baseline):
- Noclip: p = 0.0625 (Sharpe, Sortino, DD)
- Nocost/Noreset: identical metrics to baseline (no significant deviation)

---

## SHAP Interpretation

- Used `DeepExplainer` over PPO policy network
- Top regime/return features impacting policy action: e.g., `T-Bill Spread`
- See: `SHAP.png`

---

## Implementation Details

- PPO from `stable-baselines3`
- HMM for regime detection via `hmmlearn`
- SHAP interpretability via `shap`
- Vectorized training (`DummyVecEnv`)
- Horizon = 30 years per episode

---

## Recommendations

- Use `ppo_rl_stable.py` for final evaluation and production
- Use `ppo_full_pipeline.py` for training+explanation in one go
- Adjust `shock_interval`, `reset_interval`, `cost_rate` for stress tests

---

© 2025 Gabriel Nixon. All models trained on Damodaran annual data.
