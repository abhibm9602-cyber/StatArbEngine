"""
Fixed-Window Comparison: Kalman vs OLS at identical lookback windows.
Uses the EXACT preset pairs from engine.py with correct X/Y ordering.
Period: 2018-2023 formation window only.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from engine import apply_kalman_filter, apply_rolling_ols, backtest_vectorized, PRESET_PAIRS
import warnings
warnings.filterwarnings('ignore')

def run_full_fixed_window():
    # PRESET_PAIRS maps name -> (ticker_0, ticker_1)
    # engine.py uses iloc[:,0] as X (regressor) and iloc[:,1] as Y (target)
    # So ticker_0 = X, ticker_1 = Y in each preset pair
    pairs = list(PRESET_PAIRS.items())

    start_date = "2018-01-01"
    end_date = "2024-01-01"  # Formation window only
    windows = [10, 20, 40]

    # Download all tickers
    all_tickers = list(set(t for _, (a, b) in pairs for t in (a, b)))
    data = yf.download(all_tickers, start=start_date, end=end_date)
    if isinstance(data.columns, pd.MultiIndex):
        try:
            data = data['Adj Close']
        except KeyError:
            data = data['Close']
    data = data.dropna()

    results = []

    for pair_name, (ticker_0, ticker_1) in pairs:
        # Build DataFrame with ticker_0 as column 0 (X/regressor), ticker_1 as column 1 (Y/target)
        prices = pd.DataFrame({
            ticker_0: data[ticker_0],
            ticker_1: data[ticker_1]
        })

        # Kalman spread (one per pair, independent of window)
        spread_k, hr_k = apply_kalman_filter(prices, V_w_scalar=1e-5, V_e=1e-3)

        # OLS regression window is fixed at 60 days to fit parameters
        spread_o, hr_o = apply_rolling_ols(prices, window=60)

        for w in windows:
            # Backtest both at the SAME z-score lookback (w/2.0 inside backtester -> w window)
            bt_k = backtest_vectorized(
                prices, spread_k, raw_half_life=w/2.0, hedge_ratios=hr_k,
                entry_z=2.0, exit_z=0.5, transaction_bps=3.0,
                delay=2, oos_start_date=start_date
            )
            bt_o = backtest_vectorized(
                prices, spread_o, raw_half_life=w/2.0, hedge_ratios=hr_o,
                entry_z=2.0, exit_z=0.5, transaction_bps=3.0,
                delay=2, oos_start_date=start_date
            )

            results.append({
                "Pair": pair_name,
                "Tickers(X/Y)": f"{ticker_0}/{ticker_1}",
                "Window": w,
                "Kalman Sharpe": round(bt_k.sharpe_ratio, 3),
                "OLS Sharpe": round(bt_o.sharpe_ratio, 3),
            })

    df = pd.DataFrame(results)
    df.to_csv("fixed_window_full.csv", index=False)

    print("Fixed-Window Comparison (Formation 2018-2023, 3bps, shift(2))")
    print("=" * 90)
    print(df.to_string(index=False))

    # Summary
    kalman_better = sum(1 for _, row in df.iterrows() if row['Kalman Sharpe'] > row['OLS Sharpe'])
    print(f"\nKalman higher Sharpe in {kalman_better} of {len(df)} cells.")

if __name__ == "__main__":
    run_full_fixed_window()
