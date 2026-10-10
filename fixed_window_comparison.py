import numpy as np
import pandas as pd
import yfinance as yf
from engine import apply_kalman_filter, apply_rolling_ols, backtest_vectorized
import warnings
warnings.filterwarnings('ignore')

def run_fixed_window_comparison():
    print("Fixed Window Comparison: Kalman vs OLS")
    print("=" * 65)
    
    # Load Visa/Mastercard (Formation: 2018-2023)
    data = yf.download(['V', 'MA'], start="2018-01-01", end="2024-01-01")
    if isinstance(data.columns, pd.MultiIndex):
        data = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    prices_raw = data.dropna()
    prices = pd.DataFrame({'Y': prices_raw['V'], 'X': prices_raw['MA']})
    
    # 1. Kalman Spread (Default config)
    spread_kalman, hr_kalman = apply_kalman_filter(prices, V_w_scalar=1e-5, V_e=1e-3)
    
    # Compare across fixed lookbacks
    windows = [10, 20, 40]
    
    print(f"{'Method':<10} | {'Window':<6} | {'3bps Sharpe':<12}")
    print("-" * 35)
    
    for w in windows:
        # Kalman with fixed lookback (backtester uses window = raw_half_life * 2)
        # So raw_half_life = w / 2
        bt_k = backtest_vectorized(prices, spread_kalman, raw_half_life=w/2.0, hedge_ratios=hr_kalman, 
                                   entry_z=2.0, exit_z=0.5, transaction_bps=3.0, oos_start_date="2018-01-01")
        print(f"{'Kalman':<10} | {w:<6} | {bt_k.sharpe_ratio:<12.2f}")
        
        # OLS Spread with exactly the same rolling window
        spread_ols, hr_ols = apply_rolling_ols(prices, window=w)
        bt_o = backtest_vectorized(prices, spread_ols, raw_half_life=w/2.0, hedge_ratios=hr_ols,
                                   entry_z=2.0, exit_z=0.5, transaction_bps=3.0, oos_start_date="2018-01-01")
        print(f"{'OLS':<10} | {w:<6} | {bt_o.sharpe_ratio:<12.2f}")
        print("-" * 35)
        
if __name__ == "__main__":
    run_fixed_window_comparison()
