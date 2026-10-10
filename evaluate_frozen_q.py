import numpy as np
import pandas as pd
import yfinance as yf
from engine import PRESET_PAIRS, fit_ou_process_mle, get_lag1_autocorr, BacktestResult
from sweep_kalman import apply_kalman_sweep, backtest_window
import warnings
warnings.filterwarnings('ignore')

# Frozen parameter selected on 2018-2023 formation data:
# q* = 1e-4 (V_w = 1e-7, V_e = 1e-3)
# Rationale: Memory 1/g ~ 20 days, preserves physical cointegration half-life (6-11 days)
FROZEN_V_W = 1e-7
FROZEN_V_E = 1e-3

print("Evaluating Frozen Config (q* = 1e-4, V_w = 1e-7) ONCE on Holdout (2024-01-01 to 2026-10-10)")
print("=" * 95)
print(f"{'Pair':<30} | {'Eff Gain g':<10} | {'OOS AC':<8} | {'OOS HL (d)':<10} | {'OOS SR (0bps)':<14} | {'OOS SR (3bps)':<14}")
print("-" * 95)

oos_results = []

for name, tickers in PRESET_PAIRS.items():
    try:
        prices = yf.download(list(tickers), start="2018-01-01", end="2026-10-10", progress=False)
        prices = prices['Adj Close'] if 'Adj Close' in prices.columns else prices['Close']
        prices = prices[list(tickers)].dropna()
        
        # Apply frozen Kalman filter
        spread, hr, mean_g = apply_kalman_sweep(prices, FROZEN_V_W, FROZEN_V_E)
        
        # OU Half-life from formation window ONLY (frozen)
        train_spread = spread.loc[prices.index[50]:"2023-12-31"]
        ou_train = fit_ou_process_mle(train_spread.dropna())
        frozen_hl = (252 * np.log(2) / ou_train.kappa) if ou_train.kappa > 0 else 10.0
        
        # Evaluate on OOS window: 2024-01-01 onward
        oos_spread = spread.loc["2024-01-01":]
        oos_ac = get_lag1_autocorr(oos_spread)
        
        oos_sr_0 = backtest_window(prices, spread, frozen_hl, hr, transaction_bps=0.0, 
                                   start_date="2024-01-01", end_date=prices.index[-1].strftime("%Y-%m-%d"))
        oos_sr_3 = backtest_window(prices, spread, frozen_hl, hr, transaction_bps=3.0, 
                                   start_date="2024-01-01", end_date=prices.index[-1].strftime("%Y-%m-%d"))
        
        print(f"{name:<30} | {mean_g:>10.4f} | {oos_ac:>8.3f} | {frozen_hl:>10.2f} | {oos_sr_0:>14.2f} | {oos_sr_3:>14.2f}")
        
        oos_results.append({
            "Pair": name,
            "Frozen_V_w": FROZEN_V_W,
            "Effective_Gain_g": round(mean_g, 4),
            "OOS_Lag1_AC": round(oos_ac, 4),
            "Formation_OU_HL": round(frozen_hl, 2),
            "OOS_Sharpe_0bps": round(oos_sr_0, 2),
            "OOS_Sharpe_3bps": round(oos_sr_3, 2)
        })
    except Exception as e:
        print(f"Error {name}: {e}")

df_oos = pd.DataFrame(oos_results)
df_oos.to_csv("frozen_oos_results.csv", index=False)
print("=" * 95)
print("Saved to frozen_oos_results.csv")
