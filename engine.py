"""
StatArbEngine - Core Module
============================
Ornstein-Uhlenbeck Statistical Arbitrage Engine with PyTorch Neural Network.

Combines classical stochastic process modeling (OU process) with deep learning
for mean-reversion prediction in equity pairs.

Author: Abhijith Krishnan B M
"""

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. DATA PIPELINE
# ============================================================

def fetch_pair_data(
    ticker_a: str,
    ticker_b: str,
    start: str = "2021-01-01",
    end: str = "2026-09-30",
) -> pd.DataFrame:
    """Fetch adjusted close prices for a pair of equities."""
    tickers = [ticker_a, ticker_b]
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True)
    
    # Handle both multi-level and single-level column formats
    if isinstance(raw.columns, pd.MultiIndex):
        data = raw["Close"][[ticker_a, ticker_b]].dropna()
    else:
        data = raw[["Close"]].dropna()
        data.columns = [ticker_a]
    
    return data


# ============================================================
# 2. COINTEGRATION & SPREAD CONSTRUCTION
# ============================================================

@dataclass
class SpreadResult:
    """Container for spread analysis results."""
    spread: pd.Series
    hedge_ratio: pd.Series  # Dynamically tracked via Kalman Filter
    intercept: pd.Series    # Dynamically tracked via Kalman Filter
    coint_pvalue: float
    adf_pvalue: float
    is_cointegrated: bool
    half_life: float


def compute_spread(prices: pd.DataFrame) -> SpreadResult:
    """
    Compute the cointegration spread using a Kalman Filter.
    
    Replaces static OLS with a dynamic state-space model.
    Physics Analogy: 
    Instead of assuming a static relationship, we treat the hedge ratio 
    as a hidden state (like a particle's true velocity) and update our estimate 
    recursively via Bayesian inference as new noisy price observations (radar blips) arrive.
    """
    cols = prices.columns.tolist()
    X = prices[cols[0]].values
    Y = prices[cols[1]].values
    
    # Initialize Kalman Filter matrices (from scratch in NumPy)
    # Hidden state theta = [intercept, hedge_ratio]^T
    theta = np.zeros(2) 
    P = np.eye(2) * 1e-3  # Initial state covariance (uncertainty)
    
    Q = np.eye(2) * 1e-5  # Process noise (how fast the true relationship wanders)
    R = 1e-3              # Measurement noise (market microstructure noise)
    
    hedge_ratios = np.zeros(len(X))
    intercepts = np.zeros(len(X))
    spread_vals = np.zeros(len(X))
    
    for t in range(len(X)):
        # 1. Observation matrix at time t
        F = np.array([1, X[t]])
        
        # 2. Prediction step (Prior)
        P = P + Q  # Uncertainty grows between observations
        
        # 3. Measurement prediction error (Innovation)
        y_pred = F.dot(theta)
        error = Y[t] - y_pred
        
        # 4. Kalman Gain (Signal-to-Noise balancing)
        S = F.dot(P).dot(F.T) + R
        K = P.dot(F.T) / S
        
        # 5. Update step (Posterior)
        theta = theta + K * error
        P = (np.eye(2) - np.outer(K, F)).dot(P)
        
        # Store results
        intercepts[t] = theta[0]
        hedge_ratios[t] = theta[1]
        
        # Spread is the measurement error (Y - predicted Y)
        spread_vals[t] = error
        
    spread = pd.Series(spread_vals, index=prices.index, name="spread")
    hr_series = pd.Series(hedge_ratios, index=prices.index, name="hedge_ratio")
    int_series = pd.Series(intercepts, index=prices.index, name="intercept")

    # Engle-Granger cointegration test
    _, coint_pval, _ = coint(prices[cols[0]], prices[cols[1]])

    # Augmented Dickey-Fuller test on the spread (stationarity)
    adf_result = adfuller(spread.dropna())
    adf_pval = adf_result[1]

    # Half-life of mean reversion (from AR(1) fit)
    spread_lag = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()
    aligned = pd.DataFrame({"diff": spread_diff, "lag": spread_lag}).dropna()
    
    ar_model = sm.OLS(aligned["diff"], sm.add_constant(aligned["lag"])).fit()
    ar_theta = ar_model.params["lag"]
    
    if ar_theta < 0:
        half_life = -np.log(2) / np.log(1 + ar_theta)
    else:
        half_life = np.inf  

    return SpreadResult(
        spread=spread,
        hedge_ratio=hr_series,
        intercept=int_series,
        coint_pvalue=coint_pval,
        adf_pvalue=adf_pval,
        is_cointegrated=(coint_pval < 0.05),
        half_life=half_life,
    )


# ============================================================
# 3. ORNSTEIN-UHLENBECK PARAMETER ESTIMATION
# ============================================================

@dataclass
class OUParams:
    """
    Ornstein-Uhlenbeck process parameters.
    
    The OU process is the continuous-time analogue of a mean-reverting
    random walk, described by the SDE:
    
        dX(t) = κ (μ - X(t)) dt + σ dW(t)
    
    where:
        κ (kappa) = speed of mean reversion (spring constant)
        μ (mu)    = long-run equilibrium level
        σ (sigma) = volatility of the process
        W(t)      = standard Wiener process (Brownian motion)
    
    From Statistical Mechanics:
        This is identical to the Langevin equation for a Brownian
        particle in a harmonic potential V(x) = ½κ(x - μ)²,
        subject to thermal noise of intensity σ.
    """
    kappa: float   # Mean reversion speed
    mu: float      # Long-run mean
    sigma: float   # Volatility
    half_life: float  # Time to revert halfway: ln(2)/κ


def fit_ou_process(spread: pd.Series, dt: float = 1.0) -> OUParams:
    """
    Estimate OU parameters via Maximum Likelihood Estimation (MLE).
    
    For discrete observations at intervals dt, the OU process has the
    exact transition density:
    
        X(t+dt) | X(t) ~ N(μ + (X(t) - μ)·e^{-κ·dt},  σ²/(2κ)·(1 - e^{-2κ·dt}))
    
    We maximize the log-likelihood of these transitions.
    """
    x = spread.dropna().values

    def neg_log_likelihood(params):
        kappa, mu, sigma = params
        if kappa <= 0 or sigma <= 0:
            return 1e10

        n = len(x) - 1
        e_kdt = np.exp(-kappa * dt)
        var = (sigma**2 / (2 * kappa)) * (1 - np.exp(-2 * kappa * dt))

        if var <= 0:
            return 1e10

        # Predicted means for each transition
        predicted = mu + (x[:-1] - mu) * e_kdt
        residuals = x[1:] - predicted

        # Gaussian log-likelihood
        ll = -0.5 * n * np.log(2 * np.pi * var) - 0.5 * np.sum(residuals**2) / var
        return -ll

    # Initial guesses
    kappa0 = 1.0
    mu0 = np.mean(x)
    sigma0 = np.std(np.diff(x))

    result = minimize(
        neg_log_likelihood,
        x0=[kappa0, mu0, sigma0],
        method="Nelder-Mead",
        options={"maxiter": 10000, "xatol": 1e-8},
    )

    kappa, mu, sigma = result.x
    half_life = np.log(2) / max(kappa, 1e-10)

    return OUParams(kappa=kappa, mu=mu, sigma=sigma, half_life=half_life)


# ============================================================
# 4. PYTORCH SPREAD PREDICTOR (AUTOREGRESSIVE MODEL)
# ============================================================

class SpreadPredictor(nn.Module):
    """
    Autoregressive neural network for spread prediction.
    
    Architecture inspired by Karpathy's 'makemore' series:
    Given a context window of past spread values, predict the next value.
    
    This is identical to next-token prediction in an LLM, except:
    - Input: last `context_length` spread values (continuous)
    - Output: predicted next spread value (regression)
    """

    def __init__(self, context_length: int = 20, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_length, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerSpreadPredictor(nn.Module):
    """
    Self-Attention based spread predictor.
    
    A mini-Transformer (single-head attention + feedforward) applied
    to time-series spread prediction. Each time step in the context
    window is treated as a "token".
    
    From Karpathy's GPT series, but for financial time series.
    """

    def __init__(self, context_length: int = 20, d_model: int = 32, n_heads: int = 4):
        super().__init__()
        self.context_length = context_length
        self.d_model = d_model

        # Project each scalar spread value into d_model dimensions
        self.input_proj = nn.Linear(1, d_model)
        
        # Learnable positional encoding (like GPT)
        self.pos_embedding = nn.Parameter(torch.randn(1, context_length, d_model) * 0.02)

        # Single Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Output head: take the last token's representation and predict next value
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, context_length)
        x = x.unsqueeze(-1)  # (batch, context_length, 1)
        x = self.input_proj(x)  # (batch, context_length, d_model)
        x = x + self.pos_embedding  # Add positional encoding

        # Causal mask: each position can only attend to itself and earlier positions
        mask = nn.Transformer.generate_square_subsequent_mask(self.context_length)
        mask = mask.to(x.device)
        
        x = self.transformer(x, mask=mask)  # (batch, context_length, d_model)
        
        # Take the last token's output (like GPT predicting the next token)
        x = x[:, -1, :]  # (batch, d_model)
        return self.output_head(x)  # (batch, 1)


def prepare_sequences(
    spread: pd.Series, context_length: int = 20, train_ratio: float = 0.8
) -> Tuple[DataLoader, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, StandardScaler]:
    """
    Prepare autoregressive sequences from the spread series.
    
    Like Karpathy's character-level language model:
    - Input:  [s(t-20), s(t-19), ..., s(t-1)]  (context window)
    - Target: s(t)                                (next value)
    """
    values = spread.dropna().values.reshape(-1, 1)
    
    # Normalize (zero mean, unit variance)
    scaler = StandardScaler()
    values_scaled = scaler.fit_transform(values).flatten()

    X_data, y_data = [], []
    for i in range(len(values_scaled) - context_length):
        X_data.append(values_scaled[i : i + context_length])
        y_data.append(values_scaled[i + context_length])

    X = torch.tensor(np.array(X_data), dtype=torch.float32)
    y = torch.tensor(np.array(y_data), dtype=torch.float32).unsqueeze(1)

    # Train/test split
    split = int(train_ratio * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

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
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")

    return losses


# ============================================================
# 5. BACKTESTER
# ============================================================

@dataclass
class BacktestResult:
    """Container for backtest results."""
    trades: pd.DataFrame
    pnl_curve: pd.Series
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int


def backtest_ou_strategy(
    prices: pd.DataFrame,
    spread: pd.Series,
    ou_params: OUParams,
    hedge_ratios: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    transaction_bps: float = 5.0,
    slippage_bps: float = 5.0,
) -> BacktestResult:
    """
    Backtest a pairs trading strategy based on the OU model and Kalman Filter.
    
    Trading Logic (from the Physics analogy):
    - The spread is a Brownian particle in a harmonic potential.
    - When the particle is displaced far from equilibrium (|z| > entry_z),
      we expect it to snap back (mean-revert).
    - We exit when the particle returns close to equilibrium (|z| < exit_z).
      
    Market Friction:
    - Penalizes PnL using transaction costs and slippage in basis points (bps).
    """
    cols = prices.columns.tolist()
    spread_clean = spread.dropna()
    
    # Calculate rolling z-score using OU parameters
    mu = ou_params.mu
    sigma_eq = ou_params.sigma / np.sqrt(2 * ou_params.kappa)  # Equilibrium std
    z_scores = (spread_clean - mu) / sigma_eq

    position = 0  # +1 = long spread, -1 = short spread, 0 = flat
    trades = []
    entry_price_a, entry_price_b = 0, 0
    entry_hr = 0.0
    pnl_list = []
    cumulative_pnl = 0

    for i in range(len(spread_clean)):
        date = spread_clean.index[i]
        z = z_scores.iloc[i]
        price_a = prices[cols[0]].loc[date]
        price_b = prices[cols[1]].loc[date]
        current_hr = hedge_ratios.loc[date]

        if position == 0:
            # Entry signals
            if z > entry_z:
                position = -1  # Short spread: short Y, long X
                entry_price_a = price_a
                entry_price_b = price_b
                entry_hr = current_hr
                trades.append({
                    "date": date, "action": "SHORT_SPREAD",
                    "z_score": z, "price_a": price_a, "price_b": price_b, "hedge_ratio": entry_hr
                })
            elif z < -entry_z:
                position = 1  # Long spread: long Y, short X
                entry_price_a = price_a
                entry_price_b = price_b
                entry_hr = current_hr
                trades.append({
                    "date": date, "action": "LONG_SPREAD",
                    "z_score": z, "price_a": price_a, "price_b": price_b, "hedge_ratio": entry_hr
                })
        else:
            # Exit signals
            if (position == -1 and z < exit_z) or (position == 1 and z > -exit_z):
                # Gross PnL calculation
                if position == -1:
                    gross_pnl = (entry_price_b - price_b) + entry_hr * (price_a - entry_price_a)
                else:
                    gross_pnl = (price_b - entry_price_b) + entry_hr * (entry_price_a - price_a)
                
                # Market Friction Calculation
                # Total dollar volume traded (Entry + Exit for both legs)
                entry_volume = entry_price_b + (entry_hr * entry_price_a)
                exit_volume = price_b + (entry_hr * price_a)
                total_volume = entry_volume + exit_volume
                
                friction_cost = total_volume * ((transaction_bps + slippage_bps) / 10000.0)
                net_pnl = gross_pnl - friction_cost
                
                cumulative_pnl += net_pnl
                trades.append({
                    "date": date, "action": "CLOSE",
                    "z_score": z, "price_a": price_a, "price_b": price_b,
                    "pnl": net_pnl, "hedge_ratio": entry_hr, "friction_cost": friction_cost
                })
                position = 0

        pnl_list.append({"date": date, "cumulative_pnl": cumulative_pnl})

    # Build results
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    pnl_df = pd.DataFrame(pnl_list).set_index("date")
    pnl_curve = pnl_df["cumulative_pnl"]

    # Performance metrics
    if len(trades_df) > 0 and "pnl" in trades_df.columns:
        closed_trades = trades_df[trades_df["action"] == "CLOSE"]
        wins = closed_trades[closed_trades["pnl"] > 0]
        win_rate = len(wins) / max(len(closed_trades), 1)
        num_trades = len(closed_trades)
    else:
        win_rate = 0
        num_trades = 0

    # Sharpe ratio (annualized, assuming daily)
    daily_pnl = pnl_curve.diff().dropna()
    if daily_pnl.std() > 0:
        sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252)
    else:
        sharpe = 0

    # Max drawdown
    rolling_max = pnl_curve.cummax()
    drawdown = pnl_curve - rolling_max
    max_dd = drawdown.min()

    return BacktestResult(
        trades=trades_df,
        pnl_curve=pnl_curve,
        total_return=cumulative_pnl,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        win_rate=win_rate,
        num_trades=num_trades,
    )


# ============================================================
# 6. PREDEFINED PAIRS (Known cointegrated equities)
# ============================================================

PRESET_PAIRS = {
    "Visa / Mastercard": ("V", "MA"),
    "Coca-Cola / PepsiCo": ("KO", "PEP"),
    "Goldman Sachs / Morgan Stanley": ("GS", "MS"),
    "ExxonMobil / Chevron": ("XOM", "CVX"),
    "JPMorgan / Bank of America": ("JPM", "BAC"),
    "Google / Meta": ("GOOGL", "META"),
}
