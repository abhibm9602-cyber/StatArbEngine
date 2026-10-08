import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from statsmodels.tsa.stattools import adfuller, coint
from dataclasses import dataclass
from typing import Tuple, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 1. DATA
def fetch_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    return prices[tickers].dropna()  # Fix 9: Enforce column order

# 2. COINTEGRATION (Fix 3)
def check_cointegration(prices: pd.DataFrame) -> Tuple[float, float]:
    cols = prices.columns
    score, pvalue, _ = coint(prices[cols[0]], prices[cols[1]])
    return score, pvalue

def check_stationarity(spread: pd.Series) -> float:
    result = adfuller(spread.dropna())
    return result[1] # p-value

# 3. PURE NUMPY KALMAN FILTER (Fix 2 & 12 & 13)
def apply_kalman_filter(prices: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    x = prices.iloc[:, 0].values
    y = prices.iloc[:, 1].values
    
    # State: [beta, alpha]
    theta = np.zeros(2)
    P = np.eye(2) # Fix 12: non-singular
    
    V_w = (1e-5 / (1 - 1e-5)) * np.eye(2) # Process noise
    V_e = 1e-3 # Measurement noise
    
    hedge_ratios = np.zeros(len(y))
    spread_innovations = np.zeros(len(y))
    
    for t in range(len(y)):
        F = np.array([x[t], 1.0])
        P_prior = P + V_w
        
        # Innovation (Causal Spread - Fix 13)
        y_hat = np.dot(F, theta)
        e_t = y[t] - y_hat
        spread_innovations[t] = e_t
        
        # Kalman Gain
        Q_t = np.dot(np.dot(F, P_prior), F.T) + V_e
        K_t = np.dot(P_prior, F.T) / Q_t
        
        # A posteriori update
        theta = theta + K_t * e_t
        P = P_prior - np.outer(K_t, F) @ P_prior
        
        hedge_ratios[t] = theta[0]
        
    spread_series = pd.Series(spread_innovations, index=prices.index)
    hr_series = pd.Series(hedge_ratios, index=prices.index)
    return spread_series, hr_series

# 4. OU MLE
@dataclass
class OUParams:
    theta: float
    mu: float
    sigma: float
    kappa: float

def fit_ou_process_mle(spread: pd.Series) -> OUParams:
    dt = 1.0 / 252.0 # Fix 8: Annualized
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

# 5. VECTORIZED BACKTEST (Fix 1, 4, 5, 10)
@dataclass
class BacktestResult:
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
    
    # Burn in
    spread = spread.iloc[burn_in_days:]
    prices = prices.iloc[burn_in_days:]
    hedge_ratios = hedge_ratios.iloc[burn_in_days:]
    
    # Fix 1: Use OU to drive the strategy window
    half_life_days = max(5, int(252 * np.log(2) / ou_params.kappa))
    rolling_window = max(20, half_life_days * 2)
    
    rolling_mean = spread.rolling(window=rolling_window).mean()
    rolling_std = spread.rolling(window=rolling_window).std()
    z_scores = (spread - rolling_mean) / rolling_std
    z_scores = z_scores.fillna(0)
    
    signals = pd.Series(np.nan, index=z_scores.index)
    signals[z_scores < -entry_z] = 1
    signals[z_scores > entry_z] = -1
    signals[(z_scores > -exit_z) & (z_scores < exit_z)] = 0
    target_position = signals.ffill().fillna(0)
    
    actual_position = target_position.shift(1).fillna(0)
    
    diff_a = prices.iloc[:, 0].diff()
    diff_b = prices.iloc[:, 1].diff()
    daily_spread_pnl = diff_b - (hedge_ratios.shift(1) * diff_a)
    gross_mtm_pnl = actual_position * daily_spread_pnl
    
    trades = actual_position.diff().fillna(0)
    
    # Fix 4: No leakage in rolling vol
    rolling_vol = spread.rolling(20).std() / spread.expanding().std()
    rolling_vol = rolling_vol.fillna(1.0)
    
    total_friction_bps = (transaction_bps + (base_slippage_bps * rolling_vol)) / 10000.0
    friction_cost = abs(trades) * (prices.iloc[:, 1] + hedge_ratios.shift(1) * prices.iloc[:, 0]) * total_friction_bps
    
    net_daily_pnl = gross_mtm_pnl - friction_cost
    
    # Fix 5: Out of sample reporting. 
    oos_idx = int(len(net_daily_pnl) * 0.5)
    oos_pnl = net_daily_pnl.iloc[oos_idx:]
    
    cumulative_pnl = oos_pnl.cumsum()
    
    sharpe = (oos_pnl.mean() / oos_pnl.std()) * np.sqrt(252) if oos_pnl.std() > 0 else 0.0
    drawdown = cumulative_pnl - cumulative_pnl.cummax()
    
    # Fix 10: Trade count division
    num_trades = len(trades.iloc[oos_idx:][trades.iloc[oos_idx:] != 0]) // 2
    
    return BacktestResult(
        pnl_curve=cumulative_pnl, 
        total_return=cumulative_pnl.iloc[-1] if not cumulative_pnl.empty else 0, 
        sharpe_ratio=sharpe, 
        max_drawdown=drawdown.min() if not drawdown.empty else 0.0, 
        num_trades=num_trades
    )

class CausalTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=16, nhead=2, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        mask = nn.Transformer.generate_square_subsequent_mask(src.size(1)).to(src.device)
        x = self.transformer(self.embedding(src), mask=mask, is_causal=True)
        return self.decoder(x[:, -1, :])

def prepare_transformer_data(spread: pd.Series, context_length: int = 20, train_ratio: float = 0.8):
    values = spread.dropna().values.reshape(-1, 1)
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
    split_idx = int(train_ratio * len(X))
    
    return DataLoader(TensorDataset(X[:split_idx], y[:split_idx]), batch_size=32, shuffle=True), X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:], scaler

def train_model(model: nn.Module, train_loader: DataLoader, epochs: int = 100, lr: float = 1e-3) -> List[float]:
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    losses = []
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            loss = loss_fn(model(batch_X), batch_y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss / len(train_loader))
    return losses
