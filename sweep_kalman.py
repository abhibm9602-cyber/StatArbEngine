import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from statsmodels.tsa.stattools import adfuller, coint
from engine import PRESET_PAIRS, fit_ou_process_mle, get_lag1_autocorr, BacktestResult
import warnings
warnings.filterwarnings('ignore')

def apply_kalman_sweep(prices: pd.DataFrame, V_w_val: float, V_e_val: float = 1e-3):
    log_prices = np.log(prices)
    x = log_prices.iloc[:, 0].values
    y = log_prices.iloc[:, 1].values
    
    theta = np.zeros(2)
    P = np.eye(2)
    V_w = (V_w_val / (1.0 - V_w_val)) * np.eye(2)
    V_e = V_e_val
    
    hedge_ratios_elasticity = np.zeros(len(y))
    intercepts = np.zeros(len(y))
    gains = np.zeros(len(y))
    
    for t in range(len(y)):
        F = np.array([x[t], 1.0])
        P_prior = P + V_w
        y_hat = np.dot(F, theta)
        e_t = y[t] - y_hat
        
        Q_t = np.dot(np.dot(F, P_prior), F.T) + V_e
        K_t = np.dot(P_prior, F.T) / Q_t
        
        # Effective gain on fitted value: g = F P_prior F.T / Q_t
        g_t = np.dot(np.dot(F, P_prior), F.T) / Q_t
        gains[t] = g_t
        
        theta = theta + K_t * e_t
        P = P_prior - np.outer(K_t, F) @ P_prior
        
        hedge_ratios_elasticity[t] = theta[0]
        intercepts[t] = theta[1]
        
    hr_elasticity_series = pd.Series(hedge_ratios_elasticity, index=prices.index)
    int_series = pd.Series(intercepts, index=prices.index)
    
    spread_series = log_prices.iloc[:, 1] - (hr_elasticity_series * log_prices.iloc[:, 0] + int_series)
    hr_dollar = hr_elasticity_series * (prices.iloc[:, 1] / prices.iloc[:, 0])
    
    return spread_series, hr_dollar, np.mean(gains[50:])

def backtest_window(prices: pd.DataFrame, spread: pd.Series, raw_half_life: float, hedge_ratios: pd.Series, 
                     entry_z: float = 2.0, exit_z: float = 0.5, transaction_bps: float = 3.0, 
                     start_date: str = "2018-01-01", end_date: str = "2023-12-31") -> float:
    valid_hl = raw_half_life if (raw_half_life > 0 and raw_half_life < 252) else 10.0
    rolling_window = min(120, max(10, int(valid_hl * 2)))
    
    rolling_mean = spread.rolling(window=rolling_window).mean()
    rolling_std = spread.rolling(window=rolling_window).std()
    z_scores = (spread - rolling_mean) / rolling_std
    z_scores = z_scores.fillna(0)
    
    signals = pd.Series(np.nan, index=z_scores.index)
    signals[z_scores < -entry_z] = 1
    signals[z_scores > entry_z] = -1
    signals[(z_scores > -exit_z) & (z_scores < exit_z)] = 0
    target_position = signals.ffill().fillna(0)
    
    actual_position = target_position.shift(2).fillna(0)
    
    diff_a, diff_b = prices.iloc[:, 0].diff(), prices.iloc[:, 1].diff()
    daily_spread_pnl = diff_b - (hedge_ratios.shift(2) * diff_a)
    gross_mtm_pnl = actual_position * daily_spread_pnl
    trades = actual_position.diff().fillna(0)
    
    rolling_vol = spread.rolling(20).std() / spread.expanding().std()
    rolling_vol = rolling_vol.fillna(1.0)
    
    total_friction_bps = (transaction_bps + (3.0 * rolling_vol)) / 10000.0
    friction_cost = abs(trades) * (prices.iloc[:, 1] + hedge_ratios.shift(2) * prices.iloc[:, 0]) * total_friction_bps
    
    net_daily_pnl = gross_mtm_pnl - friction_cost
    
    mask = (net_daily_pnl.index >= start_date) & (net_daily_pnl.index <= end_date)
    window_pnl = net_daily_pnl[mask]
    
    sharpe = (window_pnl.mean() / window_pnl.std()) * np.sqrt(252) if window_pnl.std() > 0 else 0.0
    return sharpe

print("Running V_w / V_e Sweep on Formation Window (2018-01-01 to 2023-12-31 ONLY)")
print("Fixed V_e = 1e-3. Sweeping V_w from 1e-8 to 1e-2.")
print("=" * 95)

v_w_values = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
sweep_pairs = ["Visa / Mastercard", "Coca-Cola / PepsiCo", "Goldman Sachs / Morgan Stanley", "Placebo: Coke / Exxon"]

all_results = []

for pair_name in sweep_pairs:
    tickers = PRESET_PAIRS[pair_name]
    prices = yf.download(list(tickers), start="2018-01-01", end="2023-12-31", progress=False)
    prices = prices['Adj Close'] if 'Adj Close' in prices.columns else prices['Close']
    prices = prices[list(tickers)].dropna()
    
    print(f"\n--- {pair_name} ---")
    print(f"{'V_w':<8} | {'q = V_w/V_e':<12} | {'Eff Gain g':<10} | {'Lag-1 AC':<10} | {'OU HL (d)':<10} | {'Train SR (3bps)':<15}")
    print("-" * 75)
    
    for v_w in v_w_values:
        q = v_w / 1e-3
        spread, hr, mean_g = apply_kalman_sweep(prices, v_w, 1e-3)
        
        train_spread = spread.iloc[50:]
        ac = get_lag1_autocorr(train_spread)
        
        ou = fit_ou_process_mle(train_spread.dropna())
        raw_hl = (252 * np.log(2) / ou.kappa) if ou.kappa > 0 else 0.0
        
        sr_3bps = backtest_window(prices, spread, raw_hl, hr, transaction_bps=3.0, 
                                  start_date=prices.index[50].strftime("%Y-%m-%d"), 
                                  end_date="2023-12-31")
        
        print(f"{v_w:<8.0e} | {q:<12.1e} | {mean_g:<10.4f} | {ac:<10.3f} | {raw_hl:<10.2f} | {sr_3bps:<15.2f}")
        
        all_results.append({
            "Pair": pair_name,
            "V_w": v_w,
            "q": q,
            "Mean_Effective_Gain_g": round(mean_g, 4),
            "Lag1_Autocorr": round(ac, 4),
            "OU_HalfLife_days": round(raw_hl, 2),
            "Formation_Sharpe_3bps": round(sr_3bps, 2)
        })

df_sweep = pd.DataFrame(all_results)
df_sweep.to_csv("sweep_formation_results.csv", index=False)
print("\n" + "=" * 95)
print("Sweep saved to sweep_formation_results.csv")
