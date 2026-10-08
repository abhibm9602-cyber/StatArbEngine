# StatArbEngine

A purely mathematical Statistical Arbitrage engine for equities, built using a custom Recursive Least Squares (RLS) Kalman filter in NumPy and exact-transition Maximum Likelihood Estimation (MLE) for Ornstein-Uhlenbeck processes.

## Features & Mathematical Rigor

* **Custom NumPy Kalman Filter**: Time-varying hedge ratios computed dynamically on log-prices to ensure homoskedasticity. $V_w$ and $V_e$ are analytically bounded. The tradable spread is defined rigorously as the *posterior state* spread (inclusive of the intercept), avoiding the common pitfall of trading white-noise innovations.
* **Ornstein-Uhlenbeck (OU) Signal Generation**: Exact-transition MLE fit exclusively on a 50% formation window. The raw theoretical half-life dynamically limits the backtester's rolling Z-score window.
* **Rolling OLS Baseline**: Built-in toggle to test if the Kalman filter actually provides edge over a simple rolling 60-day Ordinary Least Squares beta.
* **Out-of-Sample (OOS) Discipline**: 
  * Engle-Granger Cointegration and ADF stationarity tests run *only* on the training data.
  * OOS Backtest covers 2021-2024.
* **Institutional-Grade Backtester**:
  * Vectorized Mark-to-Market (MTM) daily accounting.
  * Strict `shift(2)` execution: signals generated at the close of $t$ execute on the open/close of $t+1$.
  * Volatility-scaled slippage and realistic basis-point transaction costs on the absolute dollar volume of both legs.

## Performance Reality (The "No Edge" Baseline)

This project was built to test whether textbook StatArb on highly correlated mega-cap equities survives institutional execution friction. The brutal truth: **it largely does not.**

### Results (OOS: 2021-2024, 3bps Friction)
| Pair | Engle-Granger (Train) | Lag-1 Autocorr | Half-Life | Trades | OOS Sharpe (Kalman) |
|------|-----------------------|----------------|-----------|--------|---------------------|
| V / MA | **0.009** (Cointegrated) | 0.05 | <1 day | 33 | -0.20 |
| GOOGL / META | 0.058 | 0.02 | <1 day | 33 | 0.28 |
| XOM / CVX | 0.072 | 0.04 | <1 day | 35 | -0.11 |
| GS / MS | 0.310 (Noise) | 0.03 | <1 day | 34 | 0.96 |
| KO / PEP | 0.333 (Noise) | 0.03 | <1 day | 31 | 0.26 |
| JPM / BAC | 0.865 (Noise) | 0.06 | <1 day | 32 | 0.24 |
| KO / XOM (Placebo) | 0.910 (Noise) | 0.01 | <1 day | 30 | -0.45 |

### Analysis
1. **The Spread is near White-Noise:** The lag-1 autocorrelation for all posterior spreads is near zero. Consequently, the OU MLE fits massive mean-reversion speeds ($\theta$), yielding raw half-lives under 1 day. 
2. **The "Edge" is mostly Variance:** Because the true half-life is <1 day, the backtester hits the minimum fallback window (20 days) for all pairs. Trades trigger based on normal variance crossing 2.0 standard deviations, not structural mean reversion. 
3. **Statistical Insignificance:** The only pair that passes the Engle-Granger cointegration test in the formation window (V / MA) actually loses money (-0.20 Sharpe). Goldman/Morgan Stanley produced a 0.96 Sharpe, but given its 0.31 p-value and our standard error ($\approx 0.6$ for 3 years), this is statistically indistinguishable from zero.

**Conclusion:** A linear Kalman Filter on raw mega-cap daily close prices provides no statistically significant edge over a rolling OLS baseline after accounting for volatility-scaled slippage and `shift(2)` execution delay.

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```
