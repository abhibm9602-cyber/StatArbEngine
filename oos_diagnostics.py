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
    var_1 = ret.var(ddof=1)
    
    ret_lag = spread.diff(lag).dropna()
    var_k = ret_lag.var(ddof=1)
    
    return var_k / (lag * var_1)

def get_rolling_stats(spread, window=120):
    def get_beta(s):
        s_lag = s.shift(1).dropna()
        s_curr = s.loc[s_lag.index]
        if len(s_curr) < 2: return np.nan
        cov = np.cov(s_curr, s_lag, ddof=1)[0,1]
        var = np.var(s_lag, ddof=1)
        if var == 0: return np.nan
        return cov / var
    
    betas = spread.rolling(window).apply(get_beta, raw=False).dropna()
    non_reverting_share = (betas >= 1).mean() * 100
    
    valid_betas = betas[(betas > 0) & (betas < 1)]
    hls = -np.log(2) / np.log(valid_betas)
    
    return hls, non_reverting_share

def run_diagnostics():
    tickers = ["V", "MA"]
    # Burn in from 2018!
    data = yf.download(tickers, start="2018-01-01")
    if isinstance(data.columns, pd.MultiIndex):
        try:
            data = data['Adj Close']
        except KeyError:
            data = data['Close']
    data = data.dropna()
    
    prices = pd.DataFrame({'V': data['V'], 'MA': data['MA']})
    
    # Apply slow q* = 1e-4 (V_w_scalar = 1e-7) over full series
    spread_full, _ = apply_kalman_filter(prices, V_w_scalar=1e-7, V_e=1e-3)
    
    # Slice OOS
    spread_oos = spread_full.loc["2024-01-01":]
    
    # 1. ADF (Note: biased to reject on adaptive hedge, but we report it)
    adf_stat, p_val, _, _, _, _ = adfuller(spread_oos)
    
    # 2. Variance Ratio
    vr_5 = variance_ratio(spread_oos, lag=5)
    vr_10 = variance_ratio(spread_oos, lag=10)
    
    # 3. Z-score tails (percentage outside +/- 2)
    rolling_mean = spread_oos.rolling(10).mean()
    rolling_std = spread_oos.rolling(10).std()
    z_scores = ((spread_oos - rolling_mean) / rolling_std).dropna()
    pct_outside = (np.abs(z_scores) > 2.0).mean() * 100
    
    # 4. Rolling half-life range (120-day window)
    hls, nr_share = get_rolling_stats(spread_oos, window=120)
    
    print("OOS Diagnostics for V/MA at q* = 1e-4 (V_w = 1e-7)")
    print(f"ADF p-value: {p_val:.3f} (Note: biased toward rejecting for adaptive hedges)")
    print(f"Variance Ratio (lag 5): {vr_5:.3f}")
    print(f"Variance Ratio (lag 10): {vr_10:.3f}")
    print(f"Percentage of Z-scores > |2|: {pct_outside:.1f}% (Expected ~4.5%)")
    print(f"Rolling half-life (120-day): median {hls.median():.1f}d")
    print(f"Share of non-reverting windows (beta >= 1): {nr_share:.1f}%")
    
if __name__ == "__main__":
    run_diagnostics()
