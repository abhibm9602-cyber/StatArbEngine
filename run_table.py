import numpy as np
import pandas as pd
import yfinance as yf
from engine import apply_kalman_filter, apply_rolling_ols, fit_ou_process_mle, backtest_vectorized, check_cointegration, get_lag1_autocorr, PRESET_PAIRS

import warnings
warnings.filterwarnings('ignore')

print("Pair | EG p-val | Lag-1 AC | Raw HL (d) | Trades | 0bps (K) | 3bps (K) | 3bps (OLS)")
print("-" * 85)

for name, tickers in PRESET_PAIRS.items():
    try:
        prices = yf.download(list(tickers), start="2018-01-01", end="2026-10-09", progress=False)
        prices = prices['Adj Close'] if 'Adj Close' in prices.columns else prices['Close']
        prices = prices[list(tickers)].dropna()
        
        train_prices = prices.loc[:"2023-12-31"]
        if len(train_prices) < 100: continue
        
        score, pval = check_cointegration(train_prices)
        
        # Kalman
        spread, hr = apply_kalman_filter(prices)
        train_spread = spread.loc[prices.index[50]:"2023-12-31"]
        ou = fit_ou_process_mle(train_spread.dropna())
        
        raw_hl = (252 * np.log(2) / ou.kappa) if ou.kappa > 0 else 0
        ac = get_lag1_autocorr(train_spread)
        
        # OOS Start: exactly 2024-01-01
        bt_k_0 = backtest_vectorized(prices, spread, raw_hl, hr, 2.0, 0.5, 0.0, 0.0, "2024-01-01")
        bt_k_3 = backtest_vectorized(prices, spread, raw_hl, hr, 2.0, 0.5, 3.0, 3.0, "2024-01-01")
        
        # OLS Baseline
        spread_ols, hr_ols = apply_rolling_ols(prices)
        bt_ols_3 = backtest_vectorized(prices, spread_ols, raw_hl, hr_ols, 2.0, 0.5, 3.0, 3.0, "2024-01-01")
        
        print(f"{name:<15} | {pval:.3f} | {ac:>8.3f} | {raw_hl:>10.2f} | {bt_k_3.num_trades:<6} | {bt_k_0.sharpe_ratio:>8.2f} | {bt_k_3.sharpe_ratio:>8.2f} | {bt_ols_3.sharpe_ratio:>10.2f}")
    except Exception as e:
        print(f"Error {name}: {e}")
