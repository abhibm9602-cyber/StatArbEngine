# StatArbEngine

A purely mathematical Statistical Arbitrage engine for equities, built using a custom Recursive Least Squares (RLS) Kalman filter in NumPy and exact-transition Maximum Likelihood Estimation (MLE) for Ornstein-Uhlenbeck processes.

## Features & Mathematical Rigor

* **Custom NumPy Kalman Filter**: Time-varying hedge ratios computed dynamically on log-prices to ensure noise is scale-free. The tradable spread is defined rigorously as the *posterior state* spread (inclusive of the intercept), avoiding the common pitfall of trading white-noise innovations.
* **Rolling OLS Baseline**: Built-in toggle to test if the Kalman filter actually provides edge over a simple rolling 60-day Ordinary Least Squares beta.
* **Out-of-Sample (OOS) Discipline**: 
  * Engle-Granger Cointegration and ADF stationarity tests run *only* on the training data (2018-2023).
  * OOS Backtest covers 2024 to present.
* **Vectorized Mark-to-Market Backtester**:
  * Strict `shift(2)` execution: signals generated at the close of $t$ execute on the open/close of $t+1$.
  * Volatility-scaled slippage and realistic basis-point transaction costs on the absolute dollar volume of both legs.
  * *Note on hedge rebalancing*: The dollar hedge ratio ($HR_{dollar}$) changes daily with the price ratio, so the hedge rebalances implicitly. Trading costs on this daily micro-rebalancing are currently ignored.

## Performance Reality (The "No Edge" Baseline)

This project was built to test whether textbook StatArb on highly correlated mega-cap equities survives execution friction. The conclusion: **it does not.**

### Results (Formation: 2018-2023, OOS: 2024-Present, 3bps Friction)
| Pair | EG p-val (Train) | Lag-1 AC | OU Half-Life | Trades | Sharpe (Kalman) | Sharpe (OLS Baseline) |
|------|------------------|----------|--------------|--------|-----------------|-----------------------|
| V / MA | **0.001** | 0.450 | <1 day | 45 | -0.90 | 0.34 |
| KO / PEP | **0.042** | 0.571 | <1 day | 49 | -1.33 | 0.18 |
| GS / MS | **0.024** | 0.541 | <1 day | 47 | -0.55 | -0.10 |
| XOM / CVX | 0.189 | 0.612 | <1 day | 41 | -0.09 | -0.53 |
| JPM / BAC | 0.989 | 0.602 | <1 day | 38 | -0.65 | 0.22 |
| GOOGL / META | 0.752 | 0.635 | <1 day | 47 | -0.55 | -0.50 |
| KO / XOM *(Placebo)* | 0.318 | 0.684 | <1 day | 42 | -0.32 | -0.11 |

*Sharpe standard error over the ~2.5 year OOS window is approx ±0.6, meaning all results are statistically indistinguishable from zero.*

### Analysis
1. **The Kalman Filter Absorbs the Signal:** With state noise ($V_w = 1e-5$) against observation noise ($V_e = 1e-3$), the time-varying hedge ratio absorbs most of the structural mean reversion before you can trade it. The lag-1 autocorrelation of the posterior spread is only ~0.50, and the OU MLE fits massive mean-reversion speeds, driving the calculated half-life under 1 day. 
2. **Fixed Lookback:** Because the true half-life is <1 day, the backtester hits the minimum fallback window (20 days) for all pairs. The OU parameters serve purely as a diagnostic confirming that the spread decays too fast to trade profitably at a daily frequency.
3. **Statistical Insignificance:** The pairs that passed the Engle-Granger cointegration test in the formation window (V/MA, KO/PEP, GS/MS) generated highly negative OOS Sharpes under the Kalman model, actually underperforming the naive Rolling OLS baseline. The placebo pair (KO/XOM) performed similarly to the "real" pairs, confirming the lack of structural edge.

**Conclusion:** A linear Kalman Filter on log daily close prices provides no statistically significant edge over a rolling OLS baseline after friction.

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```
