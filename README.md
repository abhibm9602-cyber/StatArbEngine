# StatArbEngine

A purely mathematical Statistical Arbitrage engine for equities, built using a custom Recursive Least Squares (RLS) Kalman filter in NumPy and exact-transition Maximum Likelihood Estimation (MLE) for Ornstein-Uhlenbeck processes.

## Features & Mathematical Rigor

* **Custom NumPy Kalman Filter**: Time-varying hedge ratios computed dynamically on log-prices to ensure noise is scale-free. The tradable spread is defined rigorously as the *posterior state* spread (inclusive of the intercept), avoiding the common pitfall of trading white-noise innovations.
* **Rolling OLS Baseline**: Built-in toggle to test if the Kalman filter actually provides edge over a simple rolling 60-day Ordinary Least Squares beta.
* **Out-of-Sample (OOS) Discipline**: 
  * Engle-Granger Cointegration and ADF stationarity tests run *only* on the training data (2018-2023).
  * OOS Backtest explicitly slices at 2024-01-01 and covers 2024 to present.
* **Vectorized Mark-to-Market Backtester**:
  * Strict `shift(2)` execution: signals generated at the close of $t$ execute on the open/close of $t+1$.
  * Volatility-scaled slippage and realistic basis-point transaction costs on the absolute dollar volume of both legs.
  * *Note on hedge rebalancing*: The dollar hedge ratio ($HR_{dollar}$) changes daily with the price ratio, so the hedge rebalances implicitly. Trading costs on this daily micro-rebalancing are currently ignored.

## Performance Reality (The "No Edge" Baseline)

This project was built to test whether textbook StatArb on highly correlated mega-cap equities survives execution friction and strict out-of-sample holdout discipline. The conclusion: **it does not.**

### Results (Formation: 2018-2023, OOS: 2024-Present)
| Pair | Log EG p-val (Train) | Lag-1 AC | OU Half-Life | Trades | 0bps Sharpe (Kalman) | 3bps Sharpe (Kalman) | 3bps Sharpe (OLS) |
|------|----------------------|----------|--------------|--------|----------------------|----------------------|-------------------|
| V / MA | **0.000** | 0.450 | 0.87 days | 27 | -0.71 | -1.16 | 0.54 |
| KO / PEP | 0.031 | 0.571 | 1.24 days | 34 | -1.22 | -1.64 | 0.46 |
| GS / MS | 0.025 | 0.541 | 1.13 days | 31 | -0.62 | -0.93 | -0.22 |
| XOM / CVX | 0.175 | 0.612 | 1.41 days | 23 | 1.04 | 0.70 | -0.60 |
| JPM / BAC | 0.969 | 0.602 | 1.37 days | 25 | -0.24 | -0.58 | 0.16 |
| GOOGL / META | 0.774 | 0.635 | 1.53 days | 31 | -0.50 | -0.66 | -0.62 |
| KO / XOM *(Placebo)* | 0.499 | 0.684 | 1.82 days | 26 | -0.41 | -0.56 | -0.18 |

*(Note: Under a Bonferroni correction for 7 hypotheses, the significance threshold is $\approx 0.007$. Only V/MA is statistically cointegrated in the formation window. OOS Sharpe standard error over the ~2.5 year window is approx $\pm0.6$.)*

### Analysis
1. **The Kalman Filter Absorbs the Signal:** With state noise ($V_w = 1e-5$) against observation noise ($V_e = 1e-3$), the time-varying hedge ratio absorbs most of the structural mean reversion before you can trade it. The lag-1 autocorrelation of the posterior spread sits between 0.45 and 0.68. The placebo pair actually has higher autocorrelation than the theoretically cointegrated pairs, showing this is measuring the filter itself, not the pair's structural cointegration.
2. **Signal Decay vs. Execution Delay:** The raw calculated OU half-life sits at roughly 1-2 days for all pairs. Because the backtester enforces a strict 2-day execution delay (`shift(2)`), the signal decays entirely before the trade can be executed. This is why even the **0 bps gross Sharpe** is negative across almost all pairs. The issue is not just friction; there is fundamentally no tradable signal at a daily frequency. 
3. **Statistical Insignificance:** The only pair that passes the Engle-Granger cointegration test in the formation window (V/MA) generated a highly negative OOS Sharpe (-1.16) under the Kalman model, significantly underperforming the naive Rolling OLS baseline (0.54).

**Conclusion:** A linear Kalman Filter on log daily close prices provides no statistically significant edge over a rolling OLS baseline, as the filter parameters absorb the mean reversion and the remaining 1-day half-life signal decays before execution.

## Installation

```bash
pip install -r requirements.txt
python run_table.py
```
