import numpy as np
import pandas as pd
import yfinance as yf
from engine import PRESET_PAIRS, backtest_vectorized
import warnings
warnings.filterwarnings('ignore')

def run_static_ols_real():
    pairs = list(PRESET_PAIRS.items())
    start_date = "2018-01-01"
    
    # Download all tickers
    all_tickers = list(set(t for _, (a, b) in pairs for t in (a, b)))
    data = yf.download(all_tickers, start=start_date)
    if isinstance(data.columns, pd.MultiIndex):
        try:
            data = data['Adj Close']
        except KeyError:
            data = data['Close']
    data = data.dropna()

    results = []

    for pair_name, (ticker_x, ticker_y) in pairs:
        prices = pd.DataFrame({
            ticker_x: data[ticker_x],
            ticker_y: data[ticker_y]
        })
        
        # Split formation and OOS
        prices_form = prices.loc[:"2023-12-31"]
        
        # Fit Static OLS on formation
        log_x = np.log(prices_form.iloc[:, 0])
        log_y = np.log(prices_form.iloc[:, 1])
        
        cov_xy = np.cov(log_x, log_y)[0, 1]
        var_x = np.var(log_x)
        beta_static = cov_xy / var_x
        alpha_static = np.mean(log_y) - beta_static * np.mean(log_x)
        
        # Apply to Full Series (backtest handles OOS masking)
        full_log_x = np.log(prices.iloc[:, 0])
        full_log_y = np.log(prices.iloc[:, 1])
        spread = full_log_y - (beta_static * full_log_x + alpha_static)
        
        # Hedge ratio in dollars
        hr_dollar = beta_static * (prices.iloc[:, 1] / prices.iloc[:, 0])
        
        # Backtest (w=20 is standard 10-day half-life assumption)
        bt_3bps = backtest_vectorized(
            prices=prices,
            spread=spread,
            raw_half_life=10.0,
            hedge_ratios=hr_dollar,
            entry_z=2.0, exit_z=0.5,
            transaction_bps=3.0,
            delay=2,
            oos_start_date="2024-01-01"
        )
        
        bt_0bps = backtest_vectorized(
            prices=prices,
            spread=spread,
            raw_half_life=10.0,
            hedge_ratios=hr_dollar,
            entry_z=2.0, exit_z=0.5,
            transaction_bps=0.0,
            delay=2,
            oos_start_date="2024-01-01"
        )
        
        results.append({
            "Pair": pair_name,
            "0bps Sharpe": round(bt_0bps.sharpe_ratio, 3),
            "3bps Sharpe": round(bt_3bps.sharpe_ratio, 3),
            "Trades": bt_3bps.num_trades
        })

    df = pd.DataFrame(results)
    df.to_csv("static_ols_real_oos.csv", index=False)
    print("Static OLS OOS Evaluation (2024+, Z-score window=20)")
    print("=================================")
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_static_ols_real()
