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


## Conclusions from Empirical and Synthetic Sweeps

1. **Empirical Edge (2024+ Holdout)**: Sweeping the Kalman noise ratio  = V_w/V_e$ over 7 orders of magnitude on 2018–2023 data varies the filtered spread half-life from ~11 days to under 1 day. Freezing the optimal formation tuning at ^* = 10^{-4}$ and evaluating once on 2024 onward yields a net 3 bps Sharpe between −0.99 and +0.83 across six pairs, and +0.22 on a placebo pair. With a standard error of $\approx 0.6$, there is no statistically significant evidence of positive edge.
2. **Synthetic Validation**: The backtester was verified on planted OU spreads (50 seeds per cell). For planted half-lives of 0.9–10 days, the backtester yields a Sharpe of $\approx$ +1.0 to +1.3 at 3 bps, compared to $\approx$ −0.5 on a random-walk null.
3. **Kalman Gain Consistency**: As coded in erify_steady_state_gain.py, the empirical effective Kalman gain observed in the sweep matches the theoretical steady-state gain formula  = (-q_{eff} + \sqrt{q_{eff}^2 + 4q_{eff}})/2$ with a maximum absolute error of $\approx 0.024$.

--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
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

### Theoretical vs. Measured Gain
With $q_{eff} = (x^2 + 1) \cdot q$, where $x = \ln(\text{price}) \approx 5.3$ for Visa, the analytic steady state gain is $g = p / (1+p)$ where $p = (q_{eff} + \sqrt{q_{eff}^2 + 4q_{eff}})/2$. The sweep cleanly tracks this analytic derivation, proving that the spread's half-life is strictly governed by the filter's noise parameter.

| $q = V_w / V_e$ | Predicted $g$ | Measured $g$ (Visa) |
| :---: | :---: | :---: |
| $10^{-4}$ | 0.053 | 0.054 |
| $10^{-2}$ | 0.413 | 0.409 |
| $10^{0}$  | 0.968 | 0.967 |

### Sweep Findings on Visa / Mastercard (Formation 2018–2023)
| $V_w$ | $q = V_w / V_e$ | Effective Gain $g$ | Lag-1 AC | OU Half-Life | Formation Sharpe (3bps) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $10^{-8}$ | $10^{-5}$ | 0.0199 | 0.940 | 10.99 days | -0.29 |
| $10^{-7}$ | $10^{-4}$ | 0.0541 | 0.894 | 6.12 days | -0.56 |
| $10^{-6}$ | $10^{-3}$ | 0.1562 | 0.756 | 2.48 days | -0.26 |
| $10^{-5}$ *(Default)* | $10^{-2}$ | 0.4093 | 0.450 | 0.87 days | -0.09 |
| $10^{-4}$ | $10^{-1}$ | 0.7827 | 0.094 | 0.29 days | +0.16 |
| $10^{-3}$ | $10^{0}$ | 0.9668 | -0.072 | n/a | +0.27 |

### Sweep Findings on Placebo: Coke / Exxon (Formation 2018–2023)
| $V_w$ | $q = V_w / V_e$ | Effective Gain $g$ | Lag-1 AC | OU Half-Life | Formation Sharpe (3bps) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $10^{-7}$ | $10^{-4}$ | 0.0410 | 0.964 | 18.95 days | -0.95 |
| $10^{-5}$ *(Default)*| $10^{-2}$ | 0.3250 | 0.684 | 1.82 days | -0.13 |
| $10^{-3}$ | $10^{0}$ | 0.9426 | 0.058 | n/a | -0.15 |

### Key Mathematical Takeaways
1. **The Filter Absorbs the Spread:** When $q \ge 10^{-2}$, the effective gain $g \ge 0.40$. The filter has a memory of only 2–3 days, chasing price moves and absorbing structural cointegration. Consequently, the Kalman half-life collapses to $< 1$ day, whereas the rolling OLS spread retains the physical 7–17 day half-life.
2. **Absorption is Causally Demonstrated (Full-Pipeline Synthetic):** The V/MA sweep showed that changing $q$ didn't change profitability on *real* data. But the full-pipeline synthetic test (Section 6) shows that on a *planted* 5-day OU edge, default-q Kalman reduces oracle Sharpe from +0.97 to +0.04 (absorption), while slow-q ($10^{-5}$) recovers it to +0.82. The reason slowing $q$ doesn't help on real V/MA is not that absorption is absent --- it's that the real spread's half-life is unstable out of sample. Absorption destroys the signal at fast $q$; at slow $q$, the signal simply is not there to recover.
3. **The Lookback Window Design:** In the primary baseline table, rolling OLS (with dynamic lookback tied to $2\times$ half-life, yielding 15-34 day windows) outperformed the default Kalman filter on V/MA and KO/PEP. Testing both at fixed equal lookbacks (10, 20, 40 days) across all 7 preset pairs (`fixed_window_full_table.py`, formation 2018–2023) shows mixed results: Kalman has a higher Sharpe in 11 of 21 cells. Neither method produces a consistently tradable edge. OLS had higher Sharpe at its own dynamic lookback; at equal lookbacks the comparison is a coin flip. Full CSV in `fixed_window_full.csv`.
4. **No Hidden Edge at Lower $q$:** When choosing $q^* = 10^{-4}$ ($V_w = 10^{-7}$) on formation data to preserve the physical half-life (6–19 days), the out-of-sample validation on the 2024–present window yields a Sharpe of **-0.99** on V/MA and **-0.05** on KO/PEP, while the Placebo pair generates **+0.22**. (Note: 2024+ is a validation set, not a clean holdout, as default-q was viewed prior).
5. **Synthetic Backtester Validation (True-Spread Feed):** A Monte Carlo test (50 seeds per cell, spread volatility = 3% of a \$100 price) feeds a *known true OU spread* directly into the backtester (`synthetic_mc.py`). This bypasses the Kalman filter and hedge-ratio estimation — it tests only whether the backtester's z-score entry/exit, execution delay, and cost accounting can detect a planted edge. It does not test the full pipeline.

| Half-Life | Delay | Cost (bps) | Mean Sharpe | 5% | 95% |
| :--- | :--- | :---: | :---: | :---: | :---: |
| RW (Null) | `shift(1)` | 0.0 | -0.10 | -0.85 | 0.57 |
| RW (Null) | `shift(1)` | 3.0 | -0.46 | -1.09 | 0.15 |
| RW (Null) | `shift(2)` | 0.0 | -0.11 | -0.86 | 0.50 |
| RW (Null) | `shift(2)` | 3.0 | -0.47 | -1.07 | 0.10 |
| 0.9 days | `shift(1)` | 0.0 | +2.11 | +1.75 | +2.49 |
| 0.9 days | `shift(2)` | 3.0 | **+0.97** | +0.43 | +1.40 |
| 2.0 days | `shift(1)` | 0.0 | +1.99 | +1.64 | +2.38 |
| 2.0 days | `shift(2)` | 3.0 | **+1.34** | +0.96 | +1.82 |
| 5.0 days | `shift(1)` | 0.0 | +1.29 | +0.90 | +1.82 |
| 5.0 days | `shift(2)` | 3.0 | **+1.01** | +0.63 | +1.48 |
| 10.0 days | `shift(1)` | 0.0 | +1.28 | +0.84 | +1.80 |
| 10.0 days | `shift(2)` | 3.0 | **+1.06** | +0.61 | +1.54 |

The null control at 0 bps is centered near zero (−0.10), confirming no look-ahead bias. At 3 bps, cost drag shifts it to −0.47. All OU half-lives from 0.9 to 10 days produce statistically significant positive Sharpes at `shift(2)` and 3 bps, confirming the backtester mechanics are sound.

---

### 6. Full-Pipeline Synthetic Test: Causal Demonstration of Absorption

The true-spread test above bypasses the Kalman filter. To causally test absorption, `synthetic_pipeline.py` simulates *prices* (not spreads): $\log Y_t = \beta \cdot \log X_t + Z_t$, where $Z_t$ is OU with a known half-life. The full Kalman filter, hedge-ratio estimation, and log-to-dollar conversion all run on the synthetic prices. 50 seeds per cell, `shift(2)`, 3 bps.

**5-day half-life, constant $\beta = 1.0$:**

| Method | $q = V_w / V_e$ | Mean Sharpe | 5% | 95% |
| :--- | :---: | :---: | :---: | :---: |
| Oracle (true spread) | — | **+0.97** | +0.56 | +1.37 |
| Kalman ($V_w = 10^{-8}$) | $10^{-5}$ | **+0.82** | +0.48 | +1.23 |
| Kalman ($V_w = 10^{-7}$) | $10^{-4}$ | **+0.67** | +0.19 | +1.20 |
| Kalman ($V_w = 10^{-5}$, default) | $10^{-2}$ | **+0.04** | -0.60 | +0.61 |
| Kalman ($V_w = 10^{-3}$) | $10^{0}$ | -0.09 | -0.57 | +0.36 |
| OLS (40-day) | — | **+0.55** | -0.04 | +1.12 |
| OLS (20-day) | — | +0.31 | -0.19 | +0.78 |

**10-day half-life, constant $\beta = 1.0$:**

| Method | $q = V_w / V_e$ | Mean Sharpe | 5% | 95% |
| :--- | :---: | :---: | :---: | :---: |
| Oracle (true spread) | — | **+1.00** | +0.56 | +1.54 |
| Kalman ($V_w = 10^{-8}$) | $10^{-5}$ | **+0.77** | +0.38 | +1.24 |
| Kalman ($V_w = 10^{-7}$) | $10^{-4}$ | **+0.54** | +0.15 | +1.18 |
| Kalman ($V_w = 10^{-5}$, default) | $10^{-2}$ | -0.14 | -0.75 | +0.46 |
| Kalman ($V_w = 10^{-3}$) | $10^{0}$ | -0.33 | -0.84 | +0.27 |
| OLS (40-day) | — | +0.37 | -0.21 | +0.93 |
| OLS (20-day) | — | +0.26 | -0.29 | +0.92 |

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

# 5. Fixed-window Kalman vs OLS comparison (formation 2018-2023)
python fixed_window_full_table.py

# 6. Monte Carlo synthetic backtester validation (true-spread feed)
python synthetic_mc.py

# 7. Full-pipeline synthetic test (causal demonstration of absorption)
python synthetic_pipeline.py

# 8. Launch interactive Streamlit diagnostic dashboard
streamlit run app.py
```

