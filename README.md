# StatArbEngine

A quantitative Statistical Arbitrage research engine for equities, built using a custom Recursive Least Squares (RLS) Kalman filter in NumPy and exact-transition Maximum Likelihood Estimation (MLE) for Ornstein-Uhlenbeck processes.

## Features & Mathematical Rigor

* **Custom NumPy Kalman Filter**: Time-varying hedge ratios computed dynamically on log-prices to ensure noise is scale-free. The tradable spread is defined rigorously as the *posterior state* spread (inclusive of the intercept), avoiding the common pitfall of trading white-noise innovations.
* **Rolling OLS Baseline**: Benchmarked against a rolling 60-day Ordinary Least Squares beta, fit with its own formation-window OU half-life.
* **Pre-Specified Formation / Holdout Split**: 
  * Engle-Granger Cointegration and ADF stationarity tests run *only* on the formation window (2018-2023).
  * Out-of-sample (OOS) holdout backtest covers 2024 to present.
* **Vectorized Mark-to-Market Backtester**:
  * Strict `shift(2)` execution: signal generated at close $t$, traded at close $t+1$ (a 1-day execution delay).
  * Volatility-scaled slippage and realistic basis-point transaction costs on the absolute dollar volume of both legs.
  * *Note on hedge rebalancing*: The dollar hedge ratio ($HR_{dollar}$) changes daily with the price ratio, so the hedge rebalances implicitly. Trading costs on this daily micro-rebalancing are currently ignored.

---

## Performance Reality (The "No Edge" Baseline)

This project was built to test whether textbook StatArb on highly correlated mega-cap equities survives execution friction and strict holdout discipline. The conclusion: **no evidence of positive edge; the adaptive filter absorbs the structural cointegration signal, and remaining deviations largely decay within the 1-day execution delay.**

### 1. Main Results (Formation: 2018-2023, OOS: 2024-Present, As of 2026-10-10)

Generated directly by `python run_table.py` (saved to `results_table.csv`):

| Pair | Log EG p-val (Train) | Lag-1 AC | Kalman HL | OLS HL | Trades | 0bps Sharpe (Kalman) | 3bps Sharpe (Kalman) | 3bps Sharpe (OLS) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Visa / Mastercard** | **0.000** | 0.450 | 0.87 d | 7.68 d | 27 | -0.71 | -1.16 | -0.04 |
| **Coca-Cola / PepsiCo** | 0.031 | 0.571 | 1.24 d | 11.44 d | 34 | -1.22 | -1.64 | -0.55 |
| **Goldman Sachs / Morgan Stanley** | 0.025 | 0.541 | 1.13 d | 11.34 d | 31 | -0.62 | -0.93 | +0.21 |
| **ExxonMobil / Chevron** | 0.175 | 0.612 | 1.41 d | 11.37 d | 23 | +1.04 | +0.70 | -0.68 |
| **JPMorgan / Bank of America** | 0.969 | 0.602 | 1.37 d | 12.80 d | 25 | -0.24 | -0.58 | +0.31 |
| **Google / Meta** | 0.774 | 0.635 | 1.53 d | 17.37 d | 31 | -0.50 | -0.66 | -0.01 |
| **Placebo: Coke / Exxon** | 0.499 | 0.684 | 1.82 d | 12.68 d | 26 | -0.41 | -0.56 | +0.53 |

*(Note: Under a Bonferroni correction for 7 hypotheses, the significance threshold is $\alpha \approx 0.05 / 7 \approx 0.007$. Only V/MA is statistically cointegrated in the formation window. OOS Sharpe standard error over the ~2.5 year window is approx $\pm0.6$. The -1.64 Sharpe for KO/PEP is ~2.7 standard errors below zero, meaning it is significantly negative.)*

---

## The Mechanism: $V_w / V_e$ Noise Sweep (Formation Window 2018–2023)

To test the hypothesis that the Kalman filter acts as a high-pass filter absorbing the mean-reversion signal, we swept the state-to-measurement noise ratio $q = V_w / V_e$ across 7 orders of magnitude on the formation data (2018–2023) using `python sweep_kalman.py` (saved to `sweep_formation_results.csv`).

With fixed observation noise $V_e = 10^{-3}$, we measured the effective gain on the fitted value $g = F P^- F^T / S$, the spread's lag-1 autocorrelation, and the exact-transition OU half-life:

### Sweep Findings on Visa / Mastercard (Formation 2018–2023)
| $V_w$ | $q = V_w / V_e$ | Effective Gain $g$ | Filter Memory ($1/g$) | Lag-1 AC | OU Half-Life | Formation Sharpe (3bps) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $10^{-8}$ | $10^{-5}$ | 0.0199 | ~50 days | 0.940 | 10.99 days | -0.29 |
| $10^{-7}$ | $10^{-4}$ | 0.0541 | ~18 days | 0.894 | 6.12 days | -0.56 |
| $10^{-6}$ | $10^{-3}$ | 0.1562 | ~6.4 days | 0.756 | 2.48 days | -0.26 |
| $10^{-5}$ *(Default)* | $10^{-2}$ | 0.4093 | ~2.4 days | 0.450 | 0.87 days | -0.09 |
| $10^{-4}$ | $10^{-1}$ | 0.7827 | ~1.3 days | 0.094 | 0.29 days | +0.16 |
| $10^{-3}$ | $10^{0}$ | 0.9668 | ~1.0 day | -0.072 | 174.67 d *(diverges)*| +0.27 |
| $10^{-2}$ | $10^{1}$ | 0.9965 | ~1.0 day | -0.100 | 174.67 d *(diverges)*| +0.22 |

### Key Mathematical Takeaways
1. **The Filter Absorbs the Spread:** The posterior residual is $r_t = (1 - g)\nu_t$. When $q \ge 10^{-2}$, the effective gain $g \ge 0.40$. The filter has a memory of only 2–3 days, chasing price moves and absorbing structural cointegration. Consequently, the Kalman half-life collapses to $< 1$ day, whereas the rolling OLS spread retains the physical 7–17 day half-life.
2. **Signal Decay vs Execution Delay:** Because the Kalman spread half-life is ~1 day under default parameters, enforcing a realistic 1-day execution delay (`shift(2)`) means the signal largely decays before trade execution. This is why even 0 bps gross Sharpe is negative.
3. **No Hidden Edge at Lower $q$:** When tuning $q$ down to $10^{-4}$ ($V_w = 10^{-7}$) on formation data, memory expands to ~20 days and the physical half-life is preserved (6–11 days). However, evaluating this frozen config **once** on the 2024–present holdout (`python evaluate_frozen_q.py`) yields an OOS Sharpe of **-0.99** on V/MA and **-0.05** on KO/PEP, while the Placebo pair generates **+0.22**. The strategy exhibits no positive edge over random noise after friction.

---

## Reproducing the Experiments

All results are fully reproducible from clean environment:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Re-generate the 7-pair comparison table and CSV
python run_table.py

# 3. Re-run the V_w / V_e parameter sweep on the formation window
python sweep_kalman.py

# 4. Evaluate the frozen q* configuration on the 2024+ holdout
python evaluate_frozen_q.py

# 5. Launch interactive Streamlit diagnostic dashboard
streamlit run app.py
```
