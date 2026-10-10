import numpy as np
import pandas as pd
from engine import backtest_vectorized
import multiprocessing as mp

def run_single_null(seed):
    np.random.seed(seed)
    n_days = 750
    dt = 1.0 / 252.0
    
    # log X
    log_x = np.cumsum(np.random.normal(0, 0.015, n_days))
    beta_true = 1.0
    
    # Pure random walk spread
    target_log_std = 0.03
    # Equivalent to daily step of a 5-day OU? No, just match empirical spread vol
    z = np.cumsum(np.random.normal(0, target_log_std * np.sqrt(dt), n_days))
    
    log_y = beta_true * log_x + z
    
    X = np.exp(log_x) * 100
    Y = np.exp(log_y) * 100
    
    prices = pd.DataFrame({'X': X, 'Y': Y}, index=pd.date_range("2024-01-01", periods=n_days, freq="B"))
    
    # True spread feed (oracle on random walk)
    spread = pd.Series(z, index=prices.index)
    hr_dollar = pd.Series(beta_true * (Y / X), index=prices.index)
    
    bt = backtest_vectorized(
        prices=prices,
        spread=spread,
        raw_half_life=5.0,  # we assume 5.0 for the z-score window in the test
        hedge_ratios=hr_dollar,
        entry_z=2.0, exit_z=0.5,
        transaction_bps=3.0,
        oos_start_date="2024-01-01",
        delay=2
    )
    return bt.sharpe_ratio

def main():
    n_seeds = 500
    with mp.Pool(mp.cpu_count()) as pool:
        sharpes = list(pool.imap(run_single_null, range(n_seeds)))
        
    print("Null Distribution (Random Walk Spread, 750 days, True-Spread Feed)")
    print(f"Mean: {np.mean(sharpes):.3f}")
    print(f"5%: {np.percentile(sharpes, 5):.3f}")
    print(f"95%: {np.percentile(sharpes, 95):.3f}")
    print(f"Std Dev: {np.std(sharpes):.3f}")
    
if __name__ == "__main__":
    main()
