import numpy as np
import pandas as pd
import yfinance as yf
from engine import apply_kalman_filter, apply_rolling_ols, backtest_vectorized
import warnings
warnings.filterwarnings('ignore')

def run_full_fixed_window():
    pairs = [
        ('V', 'MA'), ('KO', 'PEP'), ('JPM', 'BAC'), 
        ('GOOG', 'GOOGL'), ('CVX', 'XOM'), ('MSFT', 'AAPL'), 
        ('KO', 'XOM') # Placebo
    ]
    
    start_date = "2018-01-01"
    end_date = "2024-01-01"
    windows = [10, 20, 40]
    
    # Download all
    tickers = list(set([t for p in pairs for t in p]))
    data = yf.download(tickers, start=start_date, end=end_date)
    if isinstance(data.columns, pd.MultiIndex):
        data = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    data = data.dropna()
    
    results = []
    
    for y_sym, x_sym in pairs:
        prices = pd.DataFrame({'X': data[x_sym], 'Y': data[y_sym]})
        
        # Kalman Default
        spread_k, hr_k = apply_kalman_filter(prices, V_w_scalar=1e-5, V_e=1e-3)
        
        for w in windows:
            # OLS
            spread_o, hr_o = apply_rolling_ols(prices, window=w)
            
            # Backtest (w = 2 * half_life constraint in backtester)
            # Both get exact same lookback inside the backtester explicitly via raw_half_life=w/2
            bt_k = backtest_vectorized(prices, spread_k, raw_half_life=w/2.0, hedge_ratios=hr_k, entry_z=2.0, exit_z=0.5, transaction_bps=3.0, delay=2, oos_start_date=start_date)
            bt_o = backtest_vectorized(prices, spread_o, raw_half_life=w/2.0, hedge_ratios=hr_o, entry_z=2.0, exit_z=0.5, transaction_bps=3.0, delay=2, oos_start_date=start_date)
            
            results.append({
                "Pair": f"{y_sym}/{x_sym}",
                "Window": w,
                "Kalman Sharpe": bt_k.sharpe_ratio,
                "OLS Sharpe": bt_o.sharpe_ratio
            })
            
    df = pd.DataFrame(results)
    df.to_csv("fixed_window_full.csv", index=False)
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_full_fixed_window()
