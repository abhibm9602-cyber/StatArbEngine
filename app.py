import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from engine import (
    fetch_data, apply_kalman_filter, apply_rolling_ols,
    fit_ou_process_mle, backtest_vectorized, check_cointegration,
    check_stationarity, get_lag1_autocorr, PRESET_PAIRS
)

st.set_page_config(page_title="StatArbEngine Dashboard", layout="wide")

# --- 1. Cached Data Ingestion (Zero Lag on Slider Tweaks) ---
@st.cache_data(ttl=3600, show_spinner=False)
def load_market_data(tickers, start_date="2018-01-01", end_date="2026-10-10"):
    return fetch_data(list(tickers), start_date, end_date)

st.title("StatArbEngine: Pairs-Trading Research Dashboard")
st.caption("Empirical State-Space Filtering, OU Maximum Likelihood Estimation & Out-of-Sample Validation")

with st.sidebar:
    st.header("1. Asset Pair & Model")
    pair_name = st.selectbox("Select Asset Pair", list(PRESET_PAIRS.keys()), index=0)
    tickers = PRESET_PAIRS[pair_name]
    baseline_ols = st.checkbox("Use Rolling OLS Baseline (Instead of Kalman)", value=False)
    
    # State-Space Noise Parameter Sweep (The Core Research Contribution)
    if not baseline_ols:
        st.markdown("**Kalman State Noise Ratio ($q = V_w / V_e$)**")
        q_options = {
            "1e-5 (Slow Filter, Preserves Physical Spread)": (1e-8, 1e-3),
            "1e-4 (Preserves 6-10d Half-Life)": (1e-7, 1e-3),
            "1e-3": (1e-6, 1e-3),
            "1e-2 (Default, Absorbs Edge)": (1e-5, 1e-3),
            "1e-1 (Fast Tracking)": (1e-4, 1e-3),
        }
        selected_q_label = st.selectbox("Filter Responsiveness", list(q_options.keys()), index=3)
        v_w, v_e = q_options[selected_q_label]
    
    st.header("2. Strategy Parameters")
    entry_z = st.slider("Entry Z-Score", 1.0, 3.5, 2.0, 0.1)
    exit_z = st.slider("Exit Z-Score", 0.0, 1.5, 0.5, 0.1)
    
    st.header("3. Friction Model")
    transaction_bps = st.number_input("Transaction Fee (bps)", value=3.0, step=0.5)
    slippage_bps = st.number_input("Base Slippage (bps, Vol-Scaled)", value=3.0, step=0.5)

# --- 2. Live Execution on Load & Slider Tweak (Instant UI Updates) ---
with st.spinner("Fetching market data and running quantitative pipeline..."):
    prices = load_market_data(tickers)
    
    # Exact Formation Window Split (2018-2023)
    train_prices = prices.loc[:"2023-12-31"]
    coint_score, coint_pval = check_cointegration(train_prices)
    
    # Spread Generation & Clean Burn-In Slicing
    if baseline_ols:
        spread, hedge_ratios = apply_rolling_ols(prices, window=60)
        train_spread = spread.loc[prices.index[60]:"2023-12-31"]
    else:
        spread, hedge_ratios = apply_kalman_filter(prices, V_w_scalar=v_w, V_e=v_e)
        train_spread = spread.loc[prices.index[50]:"2023-12-31"]
        
    adf_pval = check_stationarity(train_spread)
    lag1_ac = get_lag1_autocorr(train_spread)
    
    # Fit Continuous OU Process via Exact Transition MLE
    ou_params = fit_ou_process_mle(train_spread.dropna())
    half_life = (252.0 * np.log(2) / ou_params.kappa) if ou_params.kappa > 0 else 0.0
    
    # Execute Vectorized Backtest (OOS: 2024-01-01 to Present)
    bt = backtest_vectorized(
        prices=prices,
        spread=spread,
        raw_half_life=half_life,
        hedge_ratios=hedge_ratios,
        entry_z=entry_z,
        exit_z=exit_z,
        transaction_bps=transaction_bps,
        base_slippage_bps=slippage_bps,
        oos_start_date="2024-01-01"
    )

# --- Row 1: Formation Statistics ---
st.subheader(f"1. Formation Window Statistics (2018–2023): {pair_name}")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("EG Cointegration p-val", f"{coint_pval:.4f}", help="Significant if < 0.007 (Bonferroni)")
col2.metric("Spread ADF p-val", f"{adf_pval:.4f}")
col3.metric("Lag-1 Autocorr", f"{lag1_ac:.3f}")
col4.metric("OU Mean-Reversion (κ)", f"{ou_params.kappa:.2f}")
col5.metric("Implied Half-Life", f"{half_life:.2f} days")

# --- Row 2: Out-of-Sample Performance ---
st.subheader("2. Out-of-Sample Holdout Performance (2024–Present)")
p1, p2, p3, p4 = st.columns(4)
p1.metric("Cumulative Net PnL", f"${bt.total_return:.2f}")
p2.metric("Annualized Net Sharpe", f"{bt.sharpe_ratio:.2f}")
p3.metric("Max Drawdown", f"${bt.max_drawdown:.2f}")
p4.metric("Total Trades Executed", f"{bt.num_trades}")

# --- Row 3: 3-Panel Visual Diagnostic ---
# Rolling Z-score calculation matching engine lookback
valid_hl = half_life if (0 < half_life < 252) else 10.0
rolling_window = min(120, max(10, int(valid_hl * 2)))
rolling_mean = spread.rolling(rolling_window).mean()
rolling_std = spread.rolling(rolling_window).std()
z_score = ((spread - rolling_mean) / rolling_std).loc["2024-01-01":]

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.07,
    subplot_titles=(
        f"OOS Spread Z-Score (Lookback: {rolling_window}d) with Entry/Exit Bands",
        "Dynamic Dollar Hedge Ratio ($HR_{dollar}$)",
        "Cumulative Net Mark-to-Market PnL ($)"
    )
)

# Panel 1: Z-score & thresholds
fig.add_trace(go.Scatter(x=z_score.index, y=z_score.values, name="Z-Score", line=dict(color="#3b82f6", width=1.5)), row=1, col=1)
fig.add_hline(y=entry_z, line_dash="dash", line_color="#ef4444", annotation_text=f"+{entry_z} Entry", row=1, col=1)
fig.add_hline(y=-entry_z, line_dash="dash", line_color="#10b981", annotation_text=f"-{entry_z} Entry", row=1, col=1)
fig.add_hline(y=exit_z, line_dash="dot", line_color="#94a3b8", row=1, col=1)
fig.add_hline(y=-exit_z, line_dash="dot", line_color="#94a3b8", row=1, col=1)

# Panel 2: Hedge Ratio
fig.add_trace(go.Scatter(x=hedge_ratios.loc["2024-01-01":].index, y=hedge_ratios.loc["2024-01-01":].values, name="Hedge Ratio", line=dict(color="#f59e0b", width=1.5)), row=2, col=1)

# Panel 3: PnL Curve
pnl_color = "#10b981" if bt.total_return >= 0 else "#ef4444"
fig.add_trace(go.Scatter(x=bt.pnl_curve.index, y=bt.pnl_curve.values, name="Cumulative PnL", fill="tozeroy", line=dict(color=pnl_color, width=2)), row=3, col=1)

fig.update_layout(height=800, showlegend=False, template="plotly_white")
st.plotly_chart(fig, use_container_width=True)
