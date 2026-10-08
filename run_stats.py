import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from statsmodels.tsa.stattools import adfuller, coint
from dataclasses import dataclass
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

from engine import fetch_data, apply_kalman_filter, fit_ou_process_mle, BacktestResult, PRESET_PAIRS

def run_sim(prices, spread, ou_params, hedge_ratios, transaction_bps):
    half_life_days = max(5, int(252 * np.log(2) / ou_params.kappa)) if ou_params.kappa > 0 else 20
    rolling_window = min(120, max(20, half_life_days * 2))
    
    rolling_mean = spread.rolling(window=rolling_window).mean()
    rolling_std = spread.rolling(window=rolling_window).std()
    z_scores = (spread - rolling_mean) / rolling_std
    z_scores = z_scores.fillna(0)
    
    signals = pd.Series(np.nan, index=z_scores.index)
    signals[z_scores < -2.0] = 1
    signals[z_scores > 2.0] = -1
    signals[(z_scores > -0.5) & (z_scores < 0.5)] = 0
    target_position = signals.ffill().fillna(0)
    
    actual_position = target_position.shift(2).fillna(0)
    
    log_prices = np.log(prices)
    diff_a, diff_b = log_prices.iloc[:, 0].diff(), log_prices.iloc[:, 1].diff()
    daily_spread_pnl = diff_b - (hedge_ratios.shift(2) * diff_a)
    gross_mtm_pnl = actual_position * daily_spread_pnl
    trades = actual_position.diff().fillna(0)
    
    rolling_vol = spread.rolling(20).std() / spread.expanding().std()
    rolling_vol = rolling_vol.fillna(1.0)
    
    total_friction_bps = (transaction_bps + (3.0 * rolling_vol)) / 10000.0
    friction_cost = abs(trades) * total_friction_bps
    
    net_daily_pnl = gross_mtm_pnl - friction_cost
    
    oos_idx = int(len(net_daily_pnl) * 0.5)
    oos_pnl = net_daily_pnl.iloc[oos_idx:]
    cumulative_pnl = oos_pnl.cumsum()
    
    sharpe = (oos_pnl.mean() / oos_pnl.std()) * np.sqrt(252) if oos_pnl.std() > 0 else 0.0
    num_trades = len(trades.iloc[oos_idx:][trades.iloc[oos_idx:] != 0]) // 2
    
    return BacktestResult(pnl_curve=cumulative_pnl, total_return=cumulative_pnl.iloc[-1] if not cumulative_pnl.empty else 0, sharpe_ratio=sharpe, max_drawdown=0.0, num_trades=num_trades)

print("Pair | EG p-val | Half-life | Trades | SR (0bps) | SR (3bps) | SR (10bps)")
print("-" * 80)
for name, tickers in PRESET_PAIRS.items():
    try:
        prices = fetch_data(list(tickers), "2018-01-01", "2024-01-01")
        if len(prices) < 100: continue
        train_prices = prices.iloc[:int(len(prices)*0.5)]
        
        _, pval, _ = coint(train_prices.iloc[:,0], train_prices.iloc[:,1])
        
        spread, hr = apply_kalman_filter(prices)
        train_spread = spread.iloc[50:int(len(spread)*0.5)]
        ou = fit_ou_process_mle(train_spread.dropna())
        hl = max(5, int(252 * np.log(2) / ou.kappa)) if ou.kappa > 0 else 0
        
        bt_0 = run_sim(prices, spread, ou, hr, 0.0)
        bt_3 = run_sim(prices, spread, ou, hr, 3.0)
        bt_10 = run_sim(prices, spread, ou, hr, 10.0)
        
        print(f"{name:<15} | {pval:.3f} | {hl:<3} | {bt_0.num_trades:<4} | {bt_0.sharpe_ratio:>5.2f} | {bt_3.sharpe_ratio:>5.2f} | {bt_10.sharpe_ratio:>5.2f}")
    except Exception as e:
        print(f"Error on {name}: {e}")
