import numpy as np
import pandas as pd
import yfinance as yf
import subprocess
from datetime import datetime
from engine import apply_kalman_filter, apply_rolling_ols, fit_ou_process_mle, backtest_vectorized, check_cointegration, get_lag1_autocorr, PRESET_PAIRS

import warnings
warnings.filterwarnings('ignore')

def get_git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "UNKNOWN"

git_hash = get_git_hash()
now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print(f"StatArbEngine Table Run")
print(f"Timestamp: {now_str}")
print(f"Git Commit: {git_hash}")
print("-" * 110)
print(f"{'Pair':<30} | {'Log EG p':<8} | {'Lag-1 AC':<8} | {'K HL (d)':<8} | {'OLS HL':<8} | {'Trades':<6} | {'0bps K':<7} | {'3bps K':<7} | {'3bps OLS':<8}")
print("-" * 110)

results = []

for name, tickers in PRESET_PAIRS.items():
    try:
        prices = yf.download(list(tickers), start="2018-01-01", end="2026-10-10", progress=False)
        prices = prices['Adj Close'] if 'Adj Close' in prices.columns else prices['Close']
        prices = prices[list(tickers)].dropna()
        
        train_prices = prices.loc[:"2023-12-31"]
        if len(train_prices) < 100: continue
        
        score, pval = check_cointegration(train_prices)
        
        # Kalman Filter Spread & Hedge Ratio
        spread, hr = apply_kalman_filter(prices)
        train_spread = spread.loc[prices.index[50]:"2023-12-31"]
        ou_k = fit_ou_process_mle(train_spread.dropna())
        
        raw_hl_k = (252 * np.log(2) / ou_k.kappa) if ou_k.kappa > 0 else 0
        ac = get_lag1_autocorr(train_spread)
        
        # Backtest Kalman (OOS Start: 2024-01-01)
        bt_k_0 = backtest_vectorized(prices, spread, raw_hl_k, hr, 2.0, 0.5, 0.0, 0.0, "2024-01-01")
        bt_k_3 = backtest_vectorized(prices, spread, raw_hl_k, hr, 2.0, 0.5, 3.0, 3.0, "2024-01-01")
        
        # OLS Baseline Spread & Hedge Ratio
        spread_ols, hr_ols = apply_rolling_ols(prices)
        train_spread_ols = spread_ols.loc[prices.index[60]:"2023-12-31"]
        ou_ols = fit_ou_process_mle(train_spread_ols.dropna())
        raw_hl_ols = (252 * np.log(2) / ou_ols.kappa) if ou_ols.kappa > 0 else 0
        
        # Backtest OLS (OOS Start: 2024-01-01) with OLS's own half-life
        bt_ols_3 = backtest_vectorized(prices, spread_ols, raw_hl_ols, hr_ols, 2.0, 0.5, 3.0, 3.0, "2024-01-01")
        
        print(f"{name:<30} | {pval:>8.3f} | {ac:>8.3f} | {raw_hl_k:>8.2f} | {raw_hl_ols:>8.2f} | {bt_k_3.num_trades:>6} | {bt_k_0.sharpe_ratio:>7.2f} | {bt_k_3.sharpe_ratio:>7.2f} | {bt_ols_3.sharpe_ratio:>8.2f}")
        
        results.append({
            "Pair": name,
            "Log_EG_pval_train": round(pval, 4),
            "Lag1_AC_Kalman": round(ac, 4),
            "Kalman_HL_days": round(raw_hl_k, 2),
            "OLS_HL_days": round(raw_hl_ols, 2),
            "Trades_OOS": bt_k_3.num_trades,
            "Sharpe_0bps_Kalman": round(bt_k_0.sharpe_ratio, 2),
            "Sharpe_3bps_Kalman": round(bt_k_3.sharpe_ratio, 2),
            "Sharpe_3bps_OLS": round(bt_ols_3.sharpe_ratio, 2)
        })
    except Exception as e:
        print(f"Error {name}: {e}")

df_results = pd.DataFrame(results)
df_results.to_csv("results_table.csv", index=False)
print("-" * 110)
print(f"Saved results to results_table.csv ({len(df_results)} pairs)")
