import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from pykalman import KalmanFilter
from dataclasses import dataclass
from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. DATA FETCHING
# ============================================================

def fetch_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical adjusted close prices from Yahoo Finance."""
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    if 'Adj Close' in data.columns:
        prices = data['Adj Close']
    else:
        prices = data['Close']
    return prices.dropna()


# ============================================================
# 2. STATE-SPACE KALMAN FILTER
# ============================================================

def apply_kalman_filter(prices: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Applies a dynamic Kalman Filter to calculate a time-varying hedge ratio.
    """
    cols = prices.columns.tolist()
    y = prices[cols[1]].values
    x = prices[cols[0]].values

    # Add a constant for the intercept
    obs_mat = np.vstack([x, np.ones(len(x))]).T[:, np.newaxis]

    # Process and Measurement Noise (calibrated dynamically or fixed per asset class)
    # A robust KF allows the state (hedge ratio, intercept) to drift.
    delta = 1e-5
    trans_cov = delta / (1 - delta) * np.eye(2)
    obs_cov = 1e-3

    kf = KalmanFilter(
        n_dim_obs=1,
        n_dim_state=2,
        initial_state_mean=np.zeros(2),
        initial_state_covariance=np.ones((2, 2)),
        transition_matrices=np.eye(2),
        observation_matrices=obs_mat,
        observation_covariance=obs_cov,
        transition_covariance=trans_cov
    )

    state_means, _ = kf.filter(y)
    hedge_ratios = pd.Series(state_means[:, 0], index=prices.index)
    intercepts = pd.Series(state_means[:, 1], index=prices.index)

    # Calculate the spread: Y - (beta * X + alpha)
    spread = prices[cols[1]] - (hedge_ratios * prices[cols[0]] + intercepts)
    
    return spread, hedge_ratios


# ============================================================
# 3. STOCHASTIC MODELING: ORNSTEIN-UHLENBECK (OU)
# ============================================================

@dataclass
class OUParams:
    theta: float  # Mean reversion speed
    mu: float     # Long-term mean
    sigma: float  # Volatility (Brownian motion magnitude)
    kappa: float  # Continuous-time speed equivalent


def fit_ou_process_mle(spread: pd.Series) -> OUParams:
    """
    Calibrates the Ornstein-Uhlenbeck SDE parameters using Maximum Likelihood Estimation (MLE).
    """
    dt = 1.0 / 252.0
    x = spread.values[:-1]
    y = spread.values[1:]

    def ou_nll(params):
        theta, mu, sigma = params
        if theta <= 0 or sigma <= 0:
            return 1e10
        # Expected value
        expected = x * np.exp(-theta * dt) + mu * (1 - np.exp(-theta * dt))
        # Variance
        variance = (sigma ** 2) * (1 - np.exp(-2 * theta * dt)) / (2 * theta)
        
        # Negative Log-Likelihood
        nll = 0.5 * np.log(2 * np.pi * variance) + ((y - expected) ** 2) / (2 * variance)
        return np.sum(nll)

    # Initial guesses
    theta_guess = 1.0
    mu_guess = np.mean(spread)
    sigma_guess = np.std(spread)
    
    res = minimize(
        ou_nll,
        x0=[theta_guess, mu_guess, sigma_guess],
        bounds=((1e-5, None), (None, None), (1e-5, None)),
        method='L-BFGS-B'
    )

    theta_est, mu_est, sigma_est = res.x
    kappa_est = theta_est
    return OUParams(theta=theta_est, mu=mu_est, sigma=sigma_est, kappa=kappa_est)


# ============================================================
# 4. PYTORCH CAUSAL TRANSFORMER (NON-LINEAR MOMENTUM)
# ============================================================

class CausalTransformer(nn.Module):
    """
    Self-Attention architecture with causal masking to prevent look-ahead bias.
    """
    def __init__(self, input_dim=1, d_model=16, nhead=2, num_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        seq_len = src.size(1)
        # Create causal mask (upper triangular = -inf)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(src.device)
        
        x = self.embedding(src)
        x = self.transformer(x, mask=mask, is_causal=True)
        # We only care about predicting the next step from the last context token
        out = self.decoder(x[:, -1, :])
        return out


def prepare_transformer_data(
    spread: pd.Series,
    context_length: int = 20,
    train_ratio: float = 0.8
) -> Tuple[DataLoader, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, StandardScaler]:
    """
    Prepare data ensuring STRICT no look-ahead bias. Scaler is fit ONLY on train data.
    """
    values = spread.dropna().values.reshape(-1, 1)
    
    # Split BEFORE scaling to prevent data leakage (look-ahead bias)
    split = int(train_ratio * len(values))
    train_raw, test_raw = values[:split], values[split:]
    
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_raw).flatten()
    test_scaled = scaler.transform(test_raw).flatten()
    
    values_scaled = np.concatenate([train_scaled, test_scaled])

    X_data, y_data = [], []
    for i in range(len(values_scaled) - context_length):
        X_data.append(values_scaled[i : i + context_length])
        y_data.append(values_scaled[i + context_length])

    X = torch.tensor(np.array(X_data), dtype=torch.float32)
    y = torch.tensor(np.array(y_data), dtype=torch.float32).unsqueeze(1)

    # Train/test split indices
    split_idx = int(train_ratio * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=32, shuffle=True
    )

    return train_loader, X_train, X_test, y_train, y_test, scaler


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
) -> List[float]:
    """Train the PyTorch model. Returns list of epoch losses."""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()
    
    losses = []
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            predictions = model(batch_X)
            loss = loss_fn(predictions, batch_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)

    return losses


# ============================================================
# 5. VECTORIZED BACKTESTER (FRICTION + MTM + NO LOOK-AHEAD)
# ============================================================

@dataclass
class BacktestResult:
    """Container for backtest results."""
    pnl_curve: pd.Series
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    num_trades: int


def backtest_vectorized(
    prices: pd.DataFrame,
    spread: pd.Series,
    ou_params: OUParams,
    hedge_ratios: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    transaction_bps: float = 5.0,
    base_slippage_bps: float = 5.0,
    burn_in_days: int = 50
) -> BacktestResult:
    """
    A pure Pandas vectorized backtester.
    Fixes Look-Ahead Bias by shifting signals 1 day forward.
    Fixes Sharpe ratio by tracking daily Mark-to-Market (MTM) PnL.
    Implements Volatility-Adjusted Slippage.
    Filters out Kalman burn-in period.
    """
    cols = prices.columns.tolist()
    
    # 1. Apply Burn-In (Ignore garbage early Kalman data)
    spread = spread.iloc[burn_in_days:]
    prices = prices.iloc[burn_in_days:]
    hedge_ratios = hedge_ratios.iloc[burn_in_days:]
    
    # 2. Calculate Rolling Z-Scores (No Look-Ahead Bias)
    # Instead of using full-sample OU parameters, we use a 60-day rolling window
    rolling_mean = spread.rolling(window=60).mean()
    rolling_std = spread.rolling(window=60).std()
    z_scores = (spread - rolling_mean) / rolling_std
    z_scores = z_scores.fillna(0)
    
    # 2. Vectorized State Machine (Generate Signals)
    signals = pd.Series(np.nan, index=z_scores.index)
    signals[z_scores < -entry_z] = 1       # Long spread
    signals[z_scores > entry_z] = -1       # Short spread
    signals[(z_scores > -exit_z) & (z_scores < exit_z)] = 0  # Exit
    
    # Forward-fill to hold the position between signals
    target_position = signals.ffill().fillna(0)
    
    # 3. SHIFT EXECUTION (Prevent Look-Ahead Bias)
    # If a signal triggers today, we execute tomorrow.
    actual_position = target_position.shift(1).fillna(0)
    
    # 4. Calculate Daily Mark-to-Market PnL
    # Daily returns of Asset A and Asset B
    diff_a = prices[cols[0]].diff()
    diff_b = prices[cols[1]].diff()
    
    # Spread PnL = (Change in B) - HedgeRatio * (Change in A)
    # We use HR from previous day to avoid look-ahead
    daily_spread_pnl = diff_b - (hedge_ratios.shift(1) * diff_a)
    
    # Gross MTM PnL
    gross_mtm_pnl = actual_position * daily_spread_pnl
    
    # 5. Calculate Friction (Turnover + Volatility Adjusted Slippage)
    trades = actual_position.diff().fillna(0)
    
    # Dynamic Volatility Slippage: High vol days incur higher slippage
    rolling_vol = spread.rolling(20).std().bfill() / spread.std()
    dynamic_slippage = base_slippage_bps * rolling_vol
    total_friction_bps = (transaction_bps + dynamic_slippage) / 10000.0
    
    # We pay friction on the absolute dollar volume of trades
    trade_volume = abs(trades) * (prices[cols[1]] + hedge_ratios.shift(1) * prices[cols[0]])
    friction_cost = trade_volume * total_friction_bps
    
    # Net Daily PnL
    net_daily_pnl = gross_mtm_pnl - friction_cost
    
    # 6. Performance Metrics
    cumulative_pnl = net_daily_pnl.cumsum()
    
    total_return = cumulative_pnl.iloc[-1] if not cumulative_pnl.empty else 0
    num_actual_trades = len(trades[trades != 0])
    
    # Sharpe Ratio on Daily MTM Returns
    if net_daily_pnl.std() > 0:
        sharpe = (net_daily_pnl.mean() / net_daily_pnl.std()) * np.sqrt(252)
    else:
        sharpe = 0.0
        
    # Max Drawdown
    rolling_max = cumulative_pnl.cummax()
    drawdown = cumulative_pnl - rolling_max
    max_dd = drawdown.min() if not drawdown.empty else 0.0
    
    return BacktestResult(
        pnl_curve=cumulative_pnl,
        total_return=total_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        num_trades=num_actual_trades
    )


# ============================================================
# 6. PREDEFINED PAIRS
# ============================================================

PRESET_PAIRS = {
    "Visa / Mastercard": ("V", "MA"),
    "Coca-Cola / PepsiCo": ("KO", "PEP"),
    "Goldman Sachs / Morgan Stanley": ("GS", "MS"),
    "ExxonMobil / Chevron": ("XOM", "CVX"),
    "JPMorgan / Bank of America": ("JPM", "BAC"),
    "Google / Meta": ("GOOGL", "META"),
}
