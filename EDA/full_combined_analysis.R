# Load required libraries
library(tidyverse)
library(readr)
library(scales)
library(boot)
library(FactoMineR)
library(factoextra)

# PCA Resilience
df <- read_csv('asset_resilience_scores.csv')
features <- c('Vol_inv', 'Cumulative Return', 'CAGR')
X <- df %>% select(all_of(features)) %>% scale()
pca <- prcomp(X, center = TRUE, scale. = TRUE)
df$PCA_Resilience <- rescale(pca$x[,1])
write_csv(df, 'pca_asset_resilience.csv')

# Monte Carlo Bootstrap
df_returns <- read_csv('cleaned_real_asset_returns.csv')
percent_cols <- c('S&P 500', 'Small Cap', 'T-Bill', '10Y Treasury', 'Baa Corporate', 'Real Estate', 'Gold')
df_returns[percent_cols] <- df_returns[percent_cols] / 100
df_returns <- df_returns %>% column_to_rownames('Year')
assets <- c('S&P 500', 'Small Cap', 'Real Estate', 'Gold', 'T-Bill', 'Baa Corporate')

bootstrap_cumret <- function(data, n_iter = 1000, ci = 95) {
  boot_results <- replicate(n_iter, prod(1 + sample(data, replace = TRUE)) - 1)
  lower <- quantile(boot_results, (100 - ci) / 2 / 100)
  upper <- quantile(boot_results, 1 - (100 - ci) / 2 / 100)
  return(c(lower, upper))
}

bootstrap_results <- lapply(assets, function(asset) {
  ci_vals <- bootstrap_cumret(df_returns[[asset]])
  return(data.frame(Asset = asset, CI_Lower = ci_vals[1], CI_Upper = ci_vals[2]))
})
bootstrap_df <- bind_rows(bootstrap_results)
write_csv(bootstrap_df, 'monte_carlo_bootstrap.csv')

# Drawdown Analysis
drawdown <- function(series) {
  peak <- cummax(series)
  dd <- (series - peak) / peak
  return(dd)
}
drawdown_stats <- lapply(assets, function(asset) {
  series <- cumprod(1 + df_returns[[asset]])
  dd <- drawdown(series)
  return(data.frame(
    Asset = asset,
    Max_Drawdown = min(dd),
    Mean_Drawdown = mean(dd)
  ))
})
drawdown_df <- bind_rows(drawdown_stats)
write_csv(drawdown_df, 'drawdown_stats.csv')

# Factor Model Approximation (placeholder regression)
returns <- df_returns
factor_returns <- returns[, c('S&P 500', 'T-Bill')] # assume Fama-French-style market + risk-free
model_results <- lapply(assets, function(asset) {
  if (asset %in% colnames(factor_returns)) return(NULL)
  lm_model <- lm(returns[[asset]] ~ ., data = factor_returns)
  coef_df <- data.frame(Asset = asset, Intercept = coef(lm_model)[1], Beta_Market = coef(lm_model)[2])
  return(coef_df)
})
factor_model_df <- bind_rows(model_results)
write_csv(factor_model_df, 'factor_model_approx.csv')