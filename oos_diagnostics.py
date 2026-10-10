import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from engine import apply_kalman_filter, PRESET_PAIRS
import warnings
warnings.filterwarnings('ignore')

def variance_ratio(spread, lag=5):
    """Calculate variance ratio for a given lag. VR < 1 implies mean reversion."""
    ret = spread.diff().dropna()
    var_1 = ret.var()
    
    ret_lag = spread.diff(lag).dropna()
    var_k = ret_lag.var()
    
    return var_k / (lag * var_1)

def rolling_half_life(spread, window=120):
    def hl(s):
        s_lag = s.shift(1).dropna()
        s_curr = s.loc[s_lag.index]
        if len(s_curr) < 2: return np.nan
        beta = np.cov(s_curr, s_lag)[0,1] / np.var(s_lag)
        if beta <= 0 or beta >= 1: return np.nan
        return -np.log(2) / np.log(beta)
    return spread.rolling(window).apply(hl, raw=False)

def run_diagnostics():
    tickers = ["V", "MA"]
    data = yf.download(tickers, start="2024-01-01")
    if isinstance(data.columns, pd.MultiIndex):
        try:
            data = data['Adj Close']
        except KeyError:
            data = data['Close']
    data = data.dropna()
    
    # engine uses column 0 as X, column 1 as Y
    prices = pd.DataFrame({'V': data['V'], 'MA': data['MA']})
    
    # Apply slow q* = 1e-4 (V_w_scalar = 1e-7)
    spread, hr = apply_kalman_filter(prices, V_w_scalar=1e-7, V_e=1e-3)
    
    # 1. ADF
    adf_stat, p_val, _, _, critical_values, _ = adfuller(spread)
    
    # 2. Variance Ratio (lag 5)
    vr_5 = variance_ratio(spread, lag=5)
    vr_10 = variance_ratio(spread, lag=10)
    
    # 3. Z-score tails (percentage outside +/- 2)
    rolling_mean = spread.rolling(10).mean()
    rolling_std = spread.rolling(10).std()
    z_scores = ((spread - rolling_mean) / rolling_std).dropna()
    pct_outside = (np.abs(z_scores) > 2.0).mean() * 100
    
    # 4. Rolling half-life range
    rhl = rolling_half_life(spread, window=60).dropna()
    
    print("OOS Diagnostics for V/MA at q* = 1e-4 (V_w = 1e-7)")
    print(f"ADF p-value: {p_val:.3f}")
    print(f"Variance Ratio (lag 5): {vr_5:.3f}")
    print(f"Variance Ratio (lag 10): {vr_10:.3f}")
    print(f"Percentage of Z-scores > |2|: {pct_outside:.1f}% (Expected ~4.5%)")
    print(f"Rolling half-life (60-day): median {rhl.median():.1f}d, min {rhl.min():.1f}d, max {rhl.max():.1f}d")
    
if __name__ == "__main__":
    run_diagnostics()
