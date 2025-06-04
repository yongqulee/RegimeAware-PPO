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

| Model Variant               | Sharpe Ratio | Max Drawdown | Notes                          |
| --------------------------- | ------------ | ------------ | ------------------------------ |
| A2C (No Regime)             | -0.65        | 4.84         | Weakest performer              |
| PPO + GMM Regimes           | -0.24        | 2.29         | Stable but not optimal         |
| PPO + Transformer Regime    | -0.28        | 2.60         | Best early generalization      |
| RecurrentPPO (LSTM)         | -0.287       | 2.60         | Strongest hybrid configuration |
| RecurrentPPO (LSTM) - Final | 0.297        | 0.33         | Closest match to benchmark     |
| Benchmark Portfolio         | 0.462        | 0.313        | Outperformed all RL variants   |


### 5.2 Observations

* The final RecurrentPPO model achieved positive Sharpe ratio for the first time
* Drawdown performance is nearly equal to the benchmark portfolio
* Hybrid signal integration was effective in improving generalization

### **5.3 Statistical Validation of Regime Predictive Power**

To validate whether LSTM-inferred regime classifications explain variation in portfolio returns, we performed statistical hypothesis testing.

**Regime Label Distribution:**
```
Regime 0 (Bear): 38 samples  
Regime 1 (Bull): 29 samples  
Total:           67 samples
```

#### **Statistical Test Summary**

| **Test**        | **Value**              | **p-value** | **Interpretation**                                  |
|-----------------|------------------------|-------------|-----------------------------------------------------|
| ANOVA           | F(1, 65) = 3.231       | 0.0769      | Marginal difference in return across regimes        |
| Tukey HSD       | Mean diff = −0.0447    | 0.0769      | Weak signal, not statistically significant          |

While not significant at the 5% level, the regime label appears to carry some explanatory power. The ~4.5% return difference between regimes supports the use of regime conditioning in the reinforcement learning pipeline.

### **5.4 Economic Utility Evaluation**

To bridge machine learning performance with financial theory, we computed investor-centric utility metrics:

| **Utility Function**       | **Mean Utility** |
|---------------------------|------------------|
| CRRA (γ = 3.0)             | 0.0297           |
| CARA (α = 3.0)             | -0.9120          |

These values indicate that the RecurrentPPO (LSTM) model provides utility-aligned performance for a risk-averse investor, strengthening the financial relevance of our learned policy.

---

### **5.5 Information-Theoretic Analysis**

To evaluate whether regime signals carry explanatory power, we computed the Mutual Information between regime labels and returns:

| **Metric**                           | **Value** |
|-------------------------------------|-----------|
| Mutual Information (regime → return) | 0.1020    |

The Mutual Information score of **0.1020** shows that regime labels encode measurable information about portfolio returns, validating the relevance of regime conditioning in the agent's learning process.



---

## 6. Conclusion

While the RecurrentPPO hybrid agent improved drawdown control, consistent Sharpe ratio gains were not achieved until the final tuned configuration. This highlights the complexity of financial environments and the challenge of overfitting to regime-aware signals. Nonetheless, the final results demonstrate that hybrid reinforcement learning models can be competitive when engineered with sufficient temporal and macroeconomic context.

---

## 7. Future Work

* Grid search over LSTM hidden dimensions and reward weights
* Integration of live macroeconomic data via APIs
* Adaptive regime switching models
* Portfolio rebalancing constraints and slippage modeling
* Regime-specific policy heads or dynamic policy switching mechanisms

---

**Researcher**: Gabriel Nixon Raj
**Affiliation**: NYU Center for Data Science
