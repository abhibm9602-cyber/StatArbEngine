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
    """
    Downloads historical equity prices with explicit corporate action adjustments
    (dividends and splits) to prevent artificial spread jumps.
    """
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)
    prices = data['Close'] if 'Close' in data.columns else data
    return prices[list(tickers)].dropna()

def check_cointegration(prices: pd.DataFrame) -> Tuple[float, float]:
    """
    Two-step Engle-Granger cointegration test on log-prices.
    Matches apply_kalman_filter & apply_rolling_ols: iloc[:, 1] is dependent (Y), iloc[:, 0] is regressor (X).
    """
    log_prices = np.log(prices)
    score, pvalue, _ = coint(log_prices.iloc[:, 1], log_prices.iloc[:, 0])
    return float(score), float(pvalue)

def get_lag1_autocorr(spread: pd.Series) -> float:
    """Discrete lag-1 autocorrelation for mean-reversion assessment."""
    return float(spread.autocorr(lag=1))

def check_stationarity(spread: pd.Series) -> float:
    """Augmented Dickey-Fuller unit-root test on spread residuals."""
    result = adfuller(spread.dropna())
    return float(result[1])

def apply_kalman_filter(prices: pd.DataFrame, V_w_scalar: float = 1e-5, V_e: float = 1e-3) -> Tuple[pd.Series, pd.Series]:
    """
    Online Recursive State-Space model tracking dynamic log-price hedge ratio (beta_t)
    and intercept (alpha_t).
    
    Observation Equation:  log(P_Y,t) = [log(P_X,t), 1.0] * [beta_t, alpha_t]^T + e_t,   e_t ~ N(0, V_e)
    State Transition:      theta_t = theta_{t-1} + w_t,                                    w_t ~ N(0, V_w)
    """
    log_prices = np.log(prices)
    x = log_prices.iloc[:, 0].values
    y = log_prices.iloc[:, 1].values
    n = len(y)
    
    # Warm-start state on first observation to avoid t=0 transient spike
    theta = np.array([y[0] / x[0] if abs(x[0]) > 1e-8 else 1.0, 0.0])
    P = np.eye(2)
    V_w = (V_w_scalar / (1.0 - V_w_scalar)) * np.eye(2)
    
    hedge_ratios_elasticity = np.zeros(n)
    intercepts = np.zeros(n)
    
    for t in range(n):
        F = np.array([x[t], 1.0])
        P_prior = P + V_w
        
        y_hat = np.dot(F, theta)
        e_t = y[t] - y_hat
        
        Q_t = np.dot(np.dot(F, P_prior), F.T) + V_e
        K_t = np.dot(P_prior, F.T) / Q_t
        
        theta = theta + K_t * e_t
        
        # Joseph stabilized covariance update: guarantees P remains symmetric positive-definite
        IKF = np.eye(2) - np.outer(K_t, F)
        P = IKF @ P_prior @ IKF.T + np.outer(K_t, K_t) * V_e
        
        hedge_ratios_elasticity[t] = theta[0]
        intercepts[t] = theta[1]
        
    hr_elasticity_series = pd.Series(hedge_ratios_elasticity, index=prices.index)
    int_series = pd.Series(intercepts, index=prices.index)
    
    # Spread is the posterior state spread (inclusive of dynamic intercept)
    spread_series = log_prices.iloc[:, 1] - (hr_elasticity_series * log_prices.iloc[:, 0] + int_series)
    
    # Exact conversion from log-elasticity to dollar/share hedge ratio
    hr_dollar = hr_elasticity_series * (prices.iloc[:, 1] / prices.iloc[:, 0])
    
    return spread_series, hr_dollar

def apply_rolling_ols(prices: pd.DataFrame, window: int = 60) -> Tuple[pd.Series, pd.Series]:
    """Standard rolling OLS hedge ratio and spread baseline."""
    log_prices = np.log(prices)
    x = log_prices.iloc[:, 0]
    y = log_prices.iloc[:, 1]
    
    rolling_cov = y.rolling(window).cov(x)
    rolling_var = x.rolling(window).var()
    rolling_beta = rolling_cov / rolling_var
    rolling_alpha = y.rolling(window).mean() - rolling_beta * x.rolling(window).mean()
    
    spread = y - (rolling_beta * x + rolling_alpha)
    hr_dollar = rolling_beta * (prices.iloc[:, 1] / prices.iloc[:, 0])
    return spread.fillna(0.0), hr_dollar.fillna(0.0)

@dataclass
class OUParams:
    theta: float
    mu: float
    sigma: float
    kappa: float

def fit_ou_process_mle(spread: pd.Series) -> OUParams:
    """
    Fits continuous-time SDE: dS_t = theta * (mu - S_t) * dt + sigma * dW_t
    via exact Gaussian transition density over discrete interval dt = 1/252.
    """
    dt = 1.0 / 252.0
    s = spread.dropna().values
    x = s[:-1]
    y = s[1:]

    def ou_nll(params):
        theta, mu, sigma = params
        if theta <= 0 or sigma <= 0:
            return 1e10
        exp_th = np.exp(-theta * dt)
        expected = x * exp_th + mu * (1.0 - exp_th)
        variance = (sigma ** 2) * (1.0 - np.exp(-2.0 * theta * dt)) / (2.0 * theta)
        if variance <= 1e-15:
            return 1e10
        nll = 0.5 * np.log(2.0 * np.pi * variance) + ((y - expected) ** 2) / (2.0 * variance)
        return float(np.sum(nll))

    # Closed-form AR(1) warm-start for mean-reversion speed (theta) and diffusion (sigma)
    var_x = np.var(x)
    cov_xy = np.cov(x, y)[0, 1] if len(x) > 1 else 0.0
    phi_est = cov_xy / var_x if var_x > 1e-12 else 0.5
    phi_clamped = float(np.clip(phi_est, 0.01, 0.999))
    theta_init = float(-np.log(phi_clamped) / dt)
    mu_init = float(np.mean(s))
    
    # Exact analytical diffusion sigma estimate from AR(1) residual variance
    resid_var = float(np.var(y - (phi_clamped * x + mu_init * (1.0 - phi_clamped))))
    sigma_init = max(1e-4, float(np.sqrt(max(1e-12, resid_var * 2.0 * theta_init / (1.0 - phi_clamped ** 2)))))

    res = minimize(
        ou_nll,
        x0=[theta_init, mu_init, sigma_init],
        bounds=((1e-5, None), (None, None), (1e-5, None)),
        method='L-BFGS-B'
    )
    return OUParams(theta=float(res.x[0]), mu=float(res.x[1]), sigma=float(res.x[2]), kappa=float(res.x[0]))

@dataclass
class BacktestResult:
    pnl_curve: pd.Series
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    num_trades: int
    z_score: pd.Series = None
    rolling_window: int = 10

def backtest_vectorized(
    prices: pd.DataFrame,
    spread: pd.Series,
    raw_half_life: float,
    hedge_ratios: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    transaction_bps: float = 5.0,
    base_slippage_bps: float = 5.0,
    oos_start_date: str = "2024-01-01",
    delay: int = 2
) -> BacktestResult:
    """
    Mark-to-market backtester enforcing strict execution delay (delay=2: trade at close t+1)
    and fill-time friction evaluation across both legs with continuous rebalancing accounting.
    """
    # Lookback window clamping to prevent sample variance explosion on short half-lives
    valid_hl = raw_half_life if (0 < raw_half_life < 252) else 10.0
    rolling_window = min(120, max(10, int(valid_hl * 2)))
    
    rolling_mean = spread.rolling(window=rolling_window).mean()
    rolling_std = spread.rolling(window=rolling_window).std()
    z_scores = ((spread - rolling_mean) / rolling_std).fillna(0.0)
    
    signals = pd.Series(np.nan, index=z_scores.index)
    signals[z_scores < -entry_z] = 1.0
    signals[z_scores > entry_z] = -1.0
    signals[(z_scores > -exit_z) & (z_scores < exit_z)] = 0.0
    target_position = signals.ffill().fillna(0.0)
    
    # Execution delay: actual_position is held during day t's close-to-close return [t-1 -> t]
    actual_position = target_position.shift(delay).fillna(0.0)
    exec_hr = hedge_ratios.shift(delay).fillna(0.0)
    
    diff_x, diff_y = prices.iloc[:, 0].diff(), prices.iloc[:, 1].diff()
    daily_spread_pnl = diff_y - (exec_hr * diff_x)
    gross_mtm_pnl = actual_position * daily_spread_pnl
    
    # Exact leg-level turnover (captures entry/exit trades and dynamic hedge ratio rebalancing on leg X)
    pos_y = actual_position
    pos_x = -actual_position * exec_hr
    turnover_y = pos_y.diff().abs().fillna(0.0)
    turnover_x = pos_x.diff().abs().fillna(0.0)
    
    # Volatility-scaled slippage (causal shift(1) for execution at close t-1)
    rolling_vol = (spread.rolling(20).std() / spread.expanding().std()).fillna(1.0)
    total_friction_bps = (transaction_bps + (base_slippage_bps * rolling_vol)) / 10000.0
    exec_friction = total_friction_bps.shift(1).fillna(0.0)
    
    exec_price_y = prices.iloc[:, 1].shift(1)
    exec_price_x = prices.iloc[:, 0].shift(1)
    traded_dollar_notional = (turnover_y * exec_price_y) + (turnover_x * exec_price_x)
    
    friction_cost = (traded_dollar_notional * exec_friction).fillna(0.0)
    net_daily_pnl = gross_mtm_pnl - friction_cost
    
    # Exact Date Split
    oos_mask = net_daily_pnl.index >= oos_start_date
    oos_pnl = net_daily_pnl[oos_mask]
    cumulative_pnl = oos_pnl.cumsum()
    
    sharpe = (oos_pnl.mean() / oos_pnl.std()) * np.sqrt(252) if oos_pnl.std() > 0 else 0.0
    drawdown = cumulative_pnl - cumulative_pnl.cummax()
    
    # Count discrete round-trip unit trades in Y (accurately handles direct sign flips)
    oos_unit_trades = pos_y.diff().fillna(0.0)[oos_mask]
    num_trades = int(np.ceil(oos_unit_trades.abs().sum() / 2.0))
    
    return BacktestResult(
        pnl_curve=cumulative_pnl,
        total_return=float(cumulative_pnl.iloc[-1]) if not cumulative_pnl.empty else 0.0,
        sharpe_ratio=float(sharpe),
        max_drawdown=float(drawdown.min()) if not drawdown.empty else 0.0,
        num_trades=num_trades,
        z_score=z_scores,
        rolling_window=rolling_window
    )

PRESET_PAIRS = {
    "Visa / Mastercard": ("V", "MA"),
    "Coca-Cola / PepsiCo": ("KO", "PEP"),
    "Goldman Sachs / Morgan Stanley": ("GS", "MS"),
    "ExxonMobil / Chevron": ("XOM", "CVX"),
    "JPMorgan / Bank of America": ("JPM", "BAC"),
    "Google / Meta": ("GOOGL", "META"),
    "Placebo: Coke / Exxon": ("KO", "XOM")
}
