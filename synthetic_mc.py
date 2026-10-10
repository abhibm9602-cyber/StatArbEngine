import numpy as np
import pandas as pd
from engine import backtest_vectorized
import multiprocessing as mp
from itertools import product

def run_single_sim(args):
    seed, hl_days, delay, bps = args
    np.random.seed(seed)
    
    n_days = 1500 # ~6 years
    dt = 1.0 / 252.0
    
    # Random walk for X (Price around 100, daily vol ~1.5%)
    dx = np.random.normal(0, 1.5, n_days)
    X = 100 + np.cumsum(dx)
    
    # Spread Z
    # Target stationary std dev of spread = 3.0 (3% of price)
    target_std = 3.0
    
    z = np.zeros(n_days)
    if hl_days is None:
        # Null control: Random Walk spread
        dz = np.random.normal(0, target_std * np.sqrt(dt), n_days)
        z = np.cumsum(dz)
    else:
        # OU process
        kappa = np.log(2) / (hl_days / 252.0)
        # var(Z) = sigma^2 / 2kappa -> sigma = sqrt(var * 2 * kappa)
        sigma_z = np.sqrt((target_std**2) * 2 * kappa)
        for t in range(1, n_days):
            decay = np.exp(-kappa * dt)
            var = (sigma_z**2) * (1 - np.exp(-2 * kappa * dt)) / (2 * kappa)
            z[t] = z[t-1] * decay + np.random.normal(0, np.sqrt(var))
            
    Y = X + z
    
    prices = pd.DataFrame({'X': X, 'Y': Y}, index=pd.date_range("2018-01-01", periods=n_days, freq="B"))
    spread = pd.Series(z, index=prices.index)
    hr = pd.Series(1.0, index=prices.index) # True hedge ratio is exactly 1.0
    
    # Backtester uses rolling window = raw_half_life * 2
    raw_hl = hl_days if hl_days is not None else 20.0 # arbitrary for RW
    
    bt = backtest_vectorized(
        prices=prices,
        spread=spread,
        raw_half_life=raw_hl,
        hedge_ratios=hr,
        entry_z=2.0,
        exit_z=0.5,
        transaction_bps=bps,
        oos_start_date="2018-01-01",
        delay=delay
    )
    
    return bt.sharpe_ratio

def run_mc_grid():
    half_lives = [None, 0.9, 2.0, 5.0, 10.0]
    delays = [1, 2]
    costs = [0.0, 3.0]
    n_seeds = 50
    
    results = []
    
    # Generate all parameter combinations
    tasks = []
    for hl in half_lives:
        for d in delays:
            for bps in costs:
                for s in range(n_seeds):
                    tasks.append((s, hl, d, bps))
                    
    # Run multiprocessing
    with mp.Pool(mp.cpu_count()) as pool:
        sharpes = list(pool.imap(run_single_sim, tasks))
        
    # Aggregate
    idx = 0
    for hl in half_lives:
        for d in delays:
            for bps in costs:
                chunk = sharpes[idx:idx+n_seeds]
                idx += n_seeds
                
                mean_s = np.mean(chunk)
                p5 = np.percentile(chunk, 5)
                p95 = np.percentile(chunk, 95)
                
                hl_str = "RW (Null)" if hl is None else f"{hl}d"
                results.append({
                    "Half-Life": hl_str,
                    "Delay": f"shift({d})",
                    "Cost(bps)": bps,
                    "Mean Sharpe": mean_s,
                    "5%": p5,
                    "95%": p95
                })
                
    df = pd.DataFrame(results)
    df.to_csv("synthetic_mc_grid.csv", index=False)
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_mc_grid()
