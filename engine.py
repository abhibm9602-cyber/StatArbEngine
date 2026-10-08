import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from statsmodels.tsa.stattools import adfuller, coint
from dataclasses import dataclass
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

def fetch_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    return prices[tickers].dropna()

def check_cointegration(prices: pd.DataFrame) -> Tuple[float, float]:
    score, pvalue, _ = coint(prices.iloc[:,0], prices.iloc[:,1])
    return score, pvalue

def get_lag1_autocorr(spread: pd.Series) -> float:
    return spread.autocorr(lag=1)

def check_stationarity(spread: pd.Series) -> float:
    result = adfuller(spread.dropna())
    return result[1] 

def apply_kalman_filter(prices: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    log_prices = np.log(prices)
    x = log_prices.iloc[:, 0].values
    y = log_prices.iloc[:, 1].values
    
    theta = np.zeros(2)
    P = np.eye(2)
    V_w = (1e-5 / (1 - 1e-5)) * np.eye(2)
    V_e = 1e-3
    
    hedge_ratios_elasticity = np.zeros(len(y))
    intercepts = np.zeros(len(y))
    
    for t in range(len(y)):
        F = np.array([x[t], 1.0])
        P_prior = P + V_w
        y_hat = np.dot(F, theta)
        e_t = y[t] - y_hat
        
        Q_t = np.dot(np.dot(F, P_prior), F.T) + V_e
        K_t = np.dot(P_prior, F.T) / Q_t
        
        theta = theta + K_t * e_t
        P = P_prior - np.outer(K_t, F) @ P_prior
        
        hedge_ratios_elasticity[t] = theta[0]
        intercepts[t] = theta[1]
        
    hr_elasticity_series = pd.Series(hedge_ratios_elasticity, index=prices.index)
    int_series = pd.Series(intercepts, index=prices.index)
    
    spread_series = log_prices.iloc[:, 1] - (hr_elasticity_series * log_prices.iloc[:, 0] + int_series)
    
    # Convert elasticity to dollar hedge ratio for the backtester PnL
    hr_dollar = hr_elasticity_series * (prices.iloc[:, 1] / prices.iloc[:, 0])
    
    return spread_series, hr_dollar

def apply_rolling_ols(prices: pd.DataFrame, window: int = 60) -> Tuple[pd.Series, pd.Series]:
    log_prices = np.log(prices)
    x = log_prices.iloc[:, 0]
    y = log_prices.iloc[:, 1]
    
    rolling_cov = y.rolling(window).cov(x)
    rolling_var = x.rolling(window).var()
    rolling_beta = rolling_cov / rolling_var
    rolling_alpha = y.rolling(window).mean() - rolling_beta * x.rolling(window).mean()
    
    spread = y - (rolling_beta * x + rolling_alpha)
    hr_dollar = rolling_beta * (prices.iloc[:, 1] / prices.iloc[:, 0])
    return spread.fillna(0), hr_dollar.fillna(0)

@dataclass
class OUParams:
    theta: float
    mu: float
    sigma: float
    kappa: float

def fit_ou_process_mle(spread: pd.Series) -> OUParams:
    dt = 1.0 / 252.0
    x = spread.values[:-1]
    y = spread.values[1:]

    def ou_nll(params):
        theta, mu, sigma = params
        if theta <= 0 or sigma <= 0: return 1e10
        expected = x * np.exp(-theta * dt) + mu * (1 - np.exp(-theta * dt))
        variance = (sigma ** 2) * (1 - np.exp(-2 * theta * dt)) / (2 * theta)
        nll = 0.5 * np.log(2 * np.pi * variance) + ((y - expected) ** 2) / (2 * variance)
        return np.sum(nll)

    res = minimize(ou_nll, x0=[1.0, np.mean(spread), np.std(spread)], bounds=((1e-5, None), (None, None), (1e-5, None)), method='L-BFGS-B')
    return OUParams(theta=res.x[0], mu=res.x[1], sigma=res.x[2], kappa=res.x[0])

@dataclass
class BacktestResult:
    pnl_curve: pd.Series
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    num_trades: int

def backtest_vectorized(prices: pd.DataFrame, spread: pd.Series, raw_half_life: float, hedge_ratios: pd.Series, entry_z: float = 2.0, exit_z: float = 0.5, transaction_bps: float = 5.0, base_slippage_bps: float = 5.0) -> BacktestResult:
    
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
    
    # Dollar PnL uses dollar prices and dollar hedge ratios
    diff_a, diff_b = prices.iloc[:, 0].diff(), prices.iloc[:, 1].diff()
    daily_spread_pnl = diff_b - (hedge_ratios.shift(2) * diff_a)
    gross_mtm_pnl = actual_position * daily_spread_pnl
    trades = actual_position.diff().fillna(0)
    
    rolling_vol = spread.rolling(20).std() / spread.expanding().std()
    rolling_vol = rolling_vol.fillna(1.0)
    
    total_friction_bps = (transaction_bps + (base_slippage_bps * rolling_vol)) / 10000.0
    friction_cost = abs(trades) * (prices.iloc[:, 1] + hedge_ratios.shift(2) * prices.iloc[:, 0]) * total_friction_bps
    
    net_daily_pnl = gross_mtm_pnl - friction_cost
    
    oos_idx = int(len(net_daily_pnl) * 0.5)
    oos_pnl = net_daily_pnl.iloc[oos_idx:]
    cumulative_pnl = oos_pnl.cumsum()
    
    sharpe = (oos_pnl.mean() / oos_pnl.std()) * np.sqrt(252) if oos_pnl.std() > 0 else 0.0
    drawdown = cumulative_pnl - cumulative_pnl.cummax()
    num_trades = len(trades.iloc[oos_idx:][trades.iloc[oos_idx:] != 0]) // 2
    
    return BacktestResult(pnl_curve=cumulative_pnl, total_return=cumulative_pnl.iloc[-1] if not cumulative_pnl.empty else 0, sharpe_ratio=sharpe, max_drawdown=drawdown.min() if not drawdown.empty else 0.0, num_trades=num_trades)

PRESET_PAIRS = {
    "Visa / Mastercard": ("V", "MA"),
    "Coca-Cola / PepsiCo": ("KO", "PEP"),
    "Goldman Sachs / Morgan Stanley": ("GS", "MS"),
    "ExxonMobil / Chevron": ("XOM", "CVX"),
    "JPMorgan / Bank of America": ("JPM", "BAC"),
    "Google / Meta": ("GOOGL", "META"),
    "Placebo: Coke / Exxon": ("KO", "XOM")
}
