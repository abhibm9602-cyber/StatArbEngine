"""
Full-Pipeline Synthetic Test: Simulate prices (not spreads), run Kalman/OLS
through the entire pipeline, and compare with the true-spread oracle feed.

This is the causal test of absorption:
- If default-q Kalman falls well below the true-spread result and slower q
  recovers it, absorption is demonstrated.
- If nothing recovers it, there's a bug in the hedge construction.

All results are labeled SYNTHETIC.
"""
import numpy as np
import pandas as pd
from engine import apply_kalman_filter, apply_rolling_ols, backtest_vectorized
import multiprocessing as mp

def run_single(args):
    seed, hl_days, method, beta_drift = args
    np.random.seed(seed)

    n_days = 1500  # ~6 years
    dt = 1.0 / 252.0

    # --- Simulate log-prices ---
    # log X: random walk with daily vol ~1.5% (annualized ~24%)
    log_x = np.cumsum(np.random.normal(0, 0.015, n_days))

    # beta: constant or slowly drifting
    if beta_drift == 0.0:
        beta_true = np.full(n_days, 1.0)
    else:
        # beta drifts as a random walk with small daily steps
        beta_true = 1.0 + np.cumsum(np.random.normal(0, beta_drift, n_days))

    # OU residual with known half-life
    kappa = np.log(2) / (hl_days / 252.0)
    # Target stationary std of ~0.03 in log-price units (roughly 3% of price)
    target_log_std = 0.03
    sigma_z = np.sqrt(target_log_std**2 * 2 * kappa)

    z = np.zeros(n_days)
    for t in range(1, n_days):
        decay = np.exp(-kappa * dt)
        var = (sigma_z**2) * (1 - np.exp(-2 * kappa * dt)) / (2 * kappa)
        z[t] = z[t-1] * decay + np.random.normal(0, np.sqrt(var))

    # log Y = beta * log X + OU residual
    log_y = beta_true * log_x + z

    # Convert to prices (anchor around 100)
    X = np.exp(log_x) * 100
    Y = np.exp(log_y) * 100

    prices = pd.DataFrame({'X': X, 'Y': Y},
                          index=pd.date_range("2018-01-01", periods=n_days, freq="B"))

    # --- Run through the pipeline ---
    half_idx = n_days // 2
    oos_date = prices.index[half_idx].strftime("%Y-%m-%d")

    if method == "oracle":
        spread = pd.Series(z, index=prices.index)
        hr_dollar = pd.Series(beta_true * (Y / X), index=prices.index)
    elif method.startswith("kalman_"):
        vw = float(method.split("_")[1])
        spread, hr_dollar = apply_kalman_filter(prices, V_w_scalar=vw, V_e=1e-3)
    elif method == "ols_20":
        spread, hr_dollar = apply_rolling_ols(prices, window=20)
    elif method == "ols_40":
        spread, hr_dollar = apply_rolling_ols(prices, window=40)
    elif method == "ols_static":
        # Fit on formation window only (first half)
        train_x = log_x[:half_idx]
        train_y = log_y[:half_idx]
        cov_xy = np.cov(train_x, train_y)[0, 1]
        var_x = np.var(train_x)
        beta_static = cov_xy / var_x
        alpha_static = np.mean(train_y) - beta_static * np.mean(train_x)
        
        spread = pd.Series(log_y - (beta_static * log_x + alpha_static), index=prices.index)
        hr_dollar = pd.Series(beta_static * (Y / X), index=prices.index)
    else:
        raise ValueError(f"Unknown method: {method}")

    bt = backtest_vectorized(
        prices=prices,
        spread=spread,
        raw_half_life=hl_days,
        hedge_ratios=hr_dollar,
        entry_z=2.0,
        exit_z=0.5,
        transaction_bps=3.0,
        oos_start_date=oos_date,
        delay=2
    )

    return bt.sharpe_ratio


def main():
    half_lives = [5.0, 10.0]
    methods = [
        "oracle",
        "kalman_1e-8",   # q=1e-5: very slow, should preserve spread
        "kalman_1e-7",   # q=1e-4: slow
        "kalman_1e-5",   # q=1e-2: default
        "kalman_1e-3",   # q=1: very fast, maximum absorption
        "ols_static",
        "ols_20",
        "ols_40",
    ]
    beta_drifts = [0.0, 1e-3]  # constant beta, then 10x larger drift
    n_seeds = 50

    tasks = []
    for hl in half_lives:
        for method in methods:
            for drift in beta_drifts:
                for s in range(n_seeds):
                    tasks.append((s, hl, method, drift))

    print(f"Running {len(tasks)} simulations across {mp.cpu_count()} cores...")

    with mp.Pool(mp.cpu_count()) as pool:
        sharpes = list(pool.imap(run_single, tasks))

    # Aggregate
    results = []
    idx = 0
    for hl in half_lives:
        for method in methods:
            for drift in beta_drifts:
                chunk = sharpes[idx:idx+n_seeds]
                idx += n_seeds

                mean_s = np.mean(chunk)
                p5 = np.percentile(chunk, 5)
                p95 = np.percentile(chunk, 95)

                drift_str = "constant" if drift == 0.0 else f"drift={drift}"
                results.append({
                    "Half-Life": f"{hl}d",
                    "Method": method,
                    "Beta": drift_str,
                    "Mean Sharpe": round(mean_s, 3),
                    "5%": round(p5, 3),
                    "95%": round(p95, 3),
                })

    df = pd.DataFrame(results)
    df.to_csv("synthetic_pipeline_grid.csv", index=False)
    print("\n[SYNTHETIC] Full-Pipeline Results (shift(2), 3bps)")
    print("=" * 85)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
