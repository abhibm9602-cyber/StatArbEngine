import pandas as pd
import numpy as np
import yfinance as yf
from engine import PRESET_PAIRS

def get_mean_log_price_squared(pair):
    tickers = PRESET_PAIRS[pair]
    prices = yf.download(list(tickers), start="2018-01-01", end="2023-12-31", progress=False)
    # yfinance multi-index columns: ('Close', 'V'), etc.
    prices = prices['Close'].dropna()
    log_prices = np.log(prices)
    x = log_prices.iloc[:, 0].values
    return np.mean(x**2 + 1)

def expected_kalman_gain(q_eff):
    return (-q_eff + np.sqrt(q_eff**2 + 4*q_eff)) / 2

if __name__ == '__main__':
    df = pd.read_csv("sweep_formation_results.csv")
    
    # Calculate q_eff for each row based on the pair
    q_eff_list = []
    scaling_factors = {}
    
    for idx, row in df.iterrows():
        pair = row['Pair']
        if pair not in scaling_factors:
            scaling_factors[pair] = get_mean_log_price_squared(pair)
        
        q_base = row['q']
        q_eff = q_base * scaling_factors[pair]
        q_eff_list.append(q_eff)
        
    df['q_eff'] = q_eff_list
    df['Expected_Gain'] = expected_kalman_gain(df['q_eff'])
    df['Gain_Error'] = np.abs(df['Mean_Effective_Gain_g'] - df['Expected_Gain'])
    
    print("Steady-State Gain Comparison (Accounting for F*V_w*F.T scaling):")
    print(df[['Pair', 'q', 'q_eff', 'Mean_Effective_Gain_g', 'Expected_Gain', 'Gain_Error']].head(10))
    
    max_error = df['Gain_Error'].max()
    print(f"\nMaximum absolute gain error across sweep: {max_error:.4e}")
    
    if max_error < 0.1:
        print("\nVERIFIED: The empirical Kalman gain matches the theoretical steady-state gain formula.")
    else:
        print("\nFAILED: Large deviation from steady-state theory.")