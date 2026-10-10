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

This project was built to test whether textbook StatArb on highly correlated mega-cap equities survives execution friction and strict holdout discipline. The conclusion: **no evidence of edge on real equity pairs**. The adaptive Kalman filter absorbs structural cointegration at default noise settings, giving the baseline tables low statistical power. Relying instead on slow-$q$ and static OLS out-of-sample tests, which are proven by synthetic simulations to capture structural edges, we find no evidence of edge across the 7 pairs.

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

### 2. Static OLS Benchmark (Formation 2018-2023, OOS 2024-Present)
A static OLS hedge fit on the formation window and traded out-of-sample over 2024+. As shown in the synthetic simulations, a static hedge is optimal if the underlying cointegration relationship is strictly stable.

| Pair | 0bps Sharpe | 3bps Sharpe | Trades |
| :--- | :---: | :---: | :---: |
| **Visa / Mastercard** | -0.41 | -0.57 | 26 |
| **Coca-Cola / PepsiCo** | -0.49 | -0.59 | 25 |
| **Goldman Sachs / Morgan Stanley** | +0.90 | +0.79 | 33 |
| **ExxonMobil / Chevron** | -0.15 | -0.25 | 23 |
| **JPMorgan / Bank of America** | +0.72 | +0.61 | 27 |
| **Google / Meta** | -0.47 | -0.51 | 28 |
| **Placebo: Coke / Exxon** | +0.50 | +0.43 | 25 |

These results echo the Kalman sweep: the maximum Sharpe is +0.79 (just over 1 SE), while most pairs—including the highly integrated Visa/Mastercard—sit solidly in the negative. With the synthetic tests showing that static OLS perfectly captures a stable structural edge, its failure to do so here provides a much stronger null result that there is simply no tradable edge in these pairs.

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
2. **Absorption is Causally Demonstrated (Full-Pipeline Synthetic):** The V/MA sweep showed that changing $q$ didn't change profitability on *real* data. But the full-pipeline synthetic test (Section 6) shows that on a *planted* 5-day OU edge, default-q Kalman reduces oracle Sharpe from +0.97 to +0.02 (absorption), while slow-$q$ ($10^{-5}$) recovers it to +0.84. The reason slowing $q$ doesn't help on real pairs remains unresolved: OOS half-lives, ADF, and Variance Ratios match a ~7-day OU process, but profitability remains deeply negative, leaving it unclear if the signal simply decays or if costs (and tight thresholds) overwhelm it.
3. **The Lookback Window Design:** In the primary baseline table, rolling OLS (with dynamic lookback tied to $2\times$ half-life, yielding 15-34 day windows) outperformed the default Kalman filter on V/MA and KO/PEP. Testing both at fixed equal lookbacks (10, 20, 40 days) across all 7 preset pairs (`fixed_window_full_table.py`, formation 2018–2023) shows mixed results: Kalman has a higher Sharpe in 13 of 21 cells. Neither method produces a consistently tradable edge. OLS had higher Sharpe at its own dynamic lookback; at equal lookbacks the comparison is a coin flip. Full CSV in `fixed_window_full.csv`.
4. **No Hidden Edge at Lower $q$:** When choosing $q^* = 10^{-4}$ ($V_w = 10^{-7}$) on formation data (chosen to preserve the physical half-life of 6–19 days, not tuned for Sharpe), the out-of-sample validation on the 2024–present window yields a Sharpe of **-0.99** on V/MA and **-0.05** on KO/PEP, while the Placebo pair generates **+0.22**. (Note: 2024+ is a validation set, not a clean holdout, as default-q was viewed prior).
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

The null control at 0 bps is centered near zero (−0.10), confirming this specific setup has no look-ahead bias. At 3 bps, cost drag shifts it to −0.47. All OU half-lives from 0.9 to 10 days produce positive Sharpes in >95% of seeds at `shift(2)` and 3 bps, confirming the backtester mechanics are sound.

---

### 6. Full-Pipeline Synthetic Test: Causal Demonstration of Absorption

The true-spread test above bypasses the Kalman filter. To causally test absorption, `synthetic_pipeline.py` simulates *prices* (not spreads): $\log Y_t = \beta \cdot \log X_t + Z_t$, where $Z_t$ is OU with a known half-life. The full Kalman filter, hedge-ratio estimation, and log-to-dollar conversion all run on the synthetic prices. 50 seeds per cell, `shift(2)`, 3 bps.

**5-day half-life, constant $\beta = 1.0$:**

| Method | $\beta$ drift | $q = V_w / V_e$ | Mean Sharpe | 5% | 95% |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Oracle (true spread) | constant | — | **+0.99** | +0.16 | +1.68 |
| Kalman ($V_w = 10^{-8}$) | constant | $10^{-5}$ | **+0.84** | -0.03 | +1.53 |
| Kalman ($V_w = 10^{-7}$) | constant | $10^{-4}$ | **+0.70** | -0.04 | +1.49 |
| Kalman ($V_w = 10^{-5}$, def) | constant | $10^{-2}$ | **+0.02** | -0.69 | +0.77 |
| OLS (static formation) | constant | — | **+0.98** | +0.20 | +1.62 |
| OLS (40-day) | constant | — | **+0.58** | -0.10 | +1.30 |
| Oracle (true spread) | $10^{-3}$/day | — | **+1.04** | +0.45 | +1.63 |
| Kalman ($V_w = 10^{-8}$) | $10^{-3}$/day | $10^{-5}$ | **+0.94** | +0.19 | +1.47 |
| Kalman ($V_w = 10^{-7}$) | $10^{-3}$/day | $10^{-4}$ | **+0.80** | +0.07 | +1.41 |
| Kalman ($V_w = 10^{-5}$, def) | $10^{-3}$/day | $10^{-2}$ | **+0.07** | -0.78 | +0.85 |
| OLS (static formation) | $10^{-3}$/day | — | **+1.03** | +0.37 | +1.61 |
| OLS (40-day) | $10^{-3}$/day | — | **+0.53** | -0.06 | +1.16 |

**10-day half-life:**

| Method | $\beta$ drift | $q = V_w / V_e$ | Mean Sharpe | 5% | 95% |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Oracle (true spread) | constant | — | **+1.05** | +0.35 | +1.78 |
| Kalman ($V_w = 10^{-8}$) | constant | $10^{-5}$ | **+0.85** | +0.12 | +1.59 |
| Kalman ($V_w = 10^{-7}$) | constant | $10^{-4}$ | **+0.62** | -0.05 | +1.55 |
| Kalman ($V_w = 10^{-5}$, def) | constant | $10^{-2}$ | -0.09 | -1.08 | +0.92 |
| OLS (static formation) | constant | — | **+1.04** | +0.46 | +1.60 |
| OLS (40-day) | constant | — | +0.44 | -0.53 | +1.35 |
| Oracle (true spread) | $10^{-3}$/day | — | **+1.01** | +0.37 | +1.76 |
| Kalman ($V_w = 10^{-8}$) | $10^{-3}$/day | $10^{-5}$ | **+0.74** | +0.09 | +1.50 |
| Kalman ($V_w = 10^{-7}$) | $10^{-3}$/day | $10^{-4}$ | **+0.49** | -0.16 | +1.38 |
| Kalman ($V_w = 10^{-5}$, def) | $10^{-3}$/day | $10^{-2}$ | -0.14 | -0.83 | +0.58 |
| OLS (static formation) | $10^{-3}$/day | — | **+1.02** | +0.32 | +1.82 |
| OLS (40-day) | $10^{-3}$/day | — | +0.35 | -0.64 | +1.22 |
**Interpretation:** Default-q Kalman ($q = 10^{-2}$) reduces a ~1.0 oracle Sharpe to near zero. Slowing the filter to $q = 10^{-5}$ recovers ~85% of the oracle. **This causally demonstrates absorption:** the fast Kalman filter chases the hedge ratio so aggressively that it absorbs the mean-reverting spread into the state, leaving no signal to trade. 

Adding a rapidly drifting $\beta$ (drift $= 10^{-3}$ per day) does not drastically change the ranking over this window. A static OLS baseline perfectly matches the oracle's performance, showing that when the structural mean-reverting edge is stable (even with slow drift), a static hedge is optimal. Rolling OLS (40-day) captures only ~55% of the oracle Sharpe because the short rolling window introduces estimation noise and lag. 

**Why real pairs still earn zero at slow $q$ (Rigorous Null Comparison):** At $q = 10^{-4}$, a planted 5-day edge gives a Sharpe of +0.70 with an SD of $\approx 0.47$ across synthetic runs. In contrast, running a strict null (a true random-walk spread, $\phi=1$) through the identical pipeline over a matching 750-day window (`synthetic_null.py`) yields a mean Sharpe of -1.16 with a 95% bound at -0.32. The real V/MA result of -0.99 sits extremely comfortably inside the no-edge random-walk range, roughly 3.5 SDs below the expected mean for a true planted edge.

Furthermore, GS/MS emerges as the best pair across multiple methods (+0.83 at $q^*$, +0.79 on Static OLS). However, they all use the exact same price path, meaning this is just a single noisy draw being measured repeatedly. A best-of-seven pick from a purely zero-edge process gives roughly +0.8 SE, perfectly matching the GS/MS result. V/MA looks far more like no edge than a planted one.

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

# 5. Fixed-window Kalman vs OLS comparison (formation 2018-2023)
python fixed_window_full_table.py

# 6. Monte Carlo synthetic backtester validation (true-spread feed)
python synthetic_mc.py

# 7. Full-pipeline synthetic test (causal demonstration of absorption)
python synthetic_pipeline.py

# 8. Launch interactive Streamlit diagnostic dashboard
streamlit run app.py
```

