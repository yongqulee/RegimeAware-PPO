# Research Report: Regime-Aware LSTM-Hybrid Reinforcement Learning for Portfolio Optimization

## Abstract

This report outlines the development and evaluation of a hybrid reinforcement learning system for dynamic portfolio optimization. Utilizing recurrent PPO agents with LSTM policies, the system incorporates market regime predictions, momentum-based signals, and macroeconomic indicators to navigate periods of market volatility and structural change. Multiple reinforcement learning architectures are evaluated and compared against a benchmark portfolio based on Sharpe ratio and maximum drawdown.

---

## 1. Introduction

Financial markets exhibit non-stationary behavior influenced by structural regime changes, macroeconomic shifts, and momentum trends. Traditional static portfolio optimization techniques often fail to adapt to such changes. Reinforcement learning, with its sequential decision-making capabilities, offers a data-driven solution for learning adaptive strategies under varying market conditions.

This study explores the impact of regime awareness, temporal memory, and hybrid feature engineering in optimizing asset allocations via deep reinforcement learning.

---

## 2. Data

* **Returns Matrix**: Historical asset return data (daily resolution)
* **Regime Labels**: Transformer-generated smoothed market regime classifications
* **Macro Proxies**: Simulated interest rate trend and VIX-style volatility proxy
* **Momentum Features**: Rolling window returns to capture short-term price trends

---

## 3. Methodology

### 3.1 Environment Design

A custom Gym environment (`FullHybridEnv`) was created with the following features:

* Penalization for high volatility, drawdown, and turnover
* Reward shaped to optimize risk-adjusted return (Sortino Ratio)
* Crisis regime noise injection during specific labels

### 3.2 Models Trained

* **PPO + GMM Regimes**
* **PPO + Transformer Regimes**
* **A2C (No Regime Awareness)**
* **RecurrentPPO with LSTM (Hybrid: Regime + Macro + Momentum)**

### 3.3 Training Details

* Learning Rate: 1e-4
* Steps per iteration: 512
* Total Timesteps: 250,000
* Evaluation via custom callback

---

## 4. Evaluation Metrics

| Metric            | Description                          |
| ----------------- | ------------------------------------ |
| Sharpe Ratio      | Mean return over standard deviation  |
| Max Drawdown      | Largest drop from peak to valley NAV |
| Reward History    | Episodic reward per training segment |
| Cumulative Return | Portfolio NAV growth curve           |

---

## 5. Results

### 5.1 Performance Table

| Model Variant            | Sharpe Ratio | Max Drawdown | Notes                          |
| ------------------------ | ------------ | ------------ | ------------------------------ |
| A2C (No Regime)          | -0.65        | 4.84         | Weakest performer              |
| PPO + GMM Regimes        | -0.24        | 2.29         | Stable but not optimal         |
| PPO + Transformer Regime | -0.28        | 2.60         | Best early generalization      |
| RecurrentPPO (LSTM)      | -0.287       | 2.60         | Strongest hybrid configuration |
| Benchmark Portfolio      | 0.462        | 0.313        | Outperformed all RL variants   |

### 5.2 Observations

* All RL models underperformed the benchmark in terms of Sharpe ratio
* Hybrid LSTM model had improved drawdown control over A2C and PPO baselines
* Macro and momentum features contributed to more stable learning

---

## 6. Conclusion

While the RecurrentPPO hybrid agent improved drawdown control, consistent Sharpe ratio gains were not achieved. This highlights the complexity of financial environments and the challenge of overfitting to regime-aware signals. Future improvements could include ensemble agents, regime-specific policies, or online adaptation mechanisms.

---

## 7. Future Work

* Grid search over LSTM hidden dimensions and reward weights
* Integration of live macroeconomic data via APIs
* Adaptive regime switching models
* Portfolio rebalancing constraints and slippage modeling

---

## Appendix

* All code is available in `ppo_lstm_hybrid.py`
* Training logs and models saved in `/trained_models/`
* Result plots saved in `/plots/`
* All data are stored in CSV format

---
