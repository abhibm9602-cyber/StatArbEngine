import numpy as np
import pandas as pd
from engine import apply_kalman_filter, backtest_vectorized
import warnings
warnings.filterwarnings('ignore')

print("Synthetic Validation: Planting a 5-day Half-Life Edge")
print("=" * 65)

np.random.seed(42)
n_days = 2000
dt = 1.0 / 252.0

# 1. Random Walk X (Log prices)
dx = np.random.normal(0, 0.015, n_days)
x_log = np.cumsum(dx)

# 2. True OU process Z with exactly 5-day half-life
hl_days_true = 5.0
kappa = np.log(2) / (hl_days_true / 252.0)
sigma_z = 0.5  # Much higher variance to ensure massive signal

z = np.zeros(n_days)
for t in range(1, n_days):
    decay = np.exp(-kappa * dt)
    var = (sigma_z**2) * (1 - np.exp(-2 * kappa * dt)) / (2 * kappa)
    z[t] = z[t-1] * decay + np.random.normal(0, np.sqrt(var))

# 3. Y = X + Z (Perfect cointegration beta = 1.0)
y_log = x_log + z

# Convert to prices
prices = pd.DataFrame({
    'X': np.exp(x_log) * 100,
    'Y': np.exp(y_log) * 100
}, index=pd.date_range("2018-01-01", periods=n_days, freq="B"))

from engine import apply_rolling_ols

# 4. Apply Kalman Filter (using a slow filter to preserve the 5-day signal)
spread, hr = apply_kalman_filter(prices, V_w_scalar=1e-7, V_e=1e-3)

# Apply OLS
spread_ols, hr_ols = apply_rolling_ols(prices, window=10)

# 5. Backtest
bt_delay1 = backtest_vectorized(prices, spread, raw_half_life=5.0, hedge_ratios=hr, 
                           entry_z=2.0, exit_z=0.5, transaction_bps=3.0, oos_start_date="2018-01-01", delay=1)
bt_delay2 = backtest_vectorized(prices, spread, raw_half_life=5.0, hedge_ratios=hr, 
                           entry_z=2.0, exit_z=0.5, transaction_bps=3.0, oos_start_date="2018-01-01", delay=2)

bt_ols_d1 = backtest_vectorized(prices, spread_ols, raw_half_life=5.0, hedge_ratios=hr_ols, 
                           entry_z=2.0, exit_z=0.5, transaction_bps=3.0, oos_start_date="2018-01-01", delay=1)

print(f"Synthetic pair true half-life: {hl_days_true} days")
print(f"Kalman 3bps (Same-Close / 0-day delay): {bt_delay1.sharpe_ratio:.2f}")
print(f"Kalman 3bps (Next-Close / 1-day delay): {bt_delay2.sharpe_ratio:.2f}")
print(f"OLS 3bps (Same-Close / 0-day delay): {bt_ols_d1.sharpe_ratio:.2f}")
print(f"Number of trades (Kalman): {bt_3.num_trades}")
print("-" * 65)
print("If 3bps Sharpe > 1.0, the pipeline correctly extracts an existing edge despite a 1-day delay.")
