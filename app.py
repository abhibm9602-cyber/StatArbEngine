"""
StatArbEngine — Streamlit Dashboard
=====================================
Interactive visualization of the Ornstein-Uhlenbeck Statistical Arbitrage Engine.

Displays:
1. Price series & cointegration analysis
2. OU process fit with SDE parameters
3. PyTorch model training & predictions
4. Backtest PnL and performance metrics

Author: Abhijith Krishnan B M
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

from engine import (
    fetch_pair_data,
    compute_spread,
    fit_ou_process,
    SpreadPredictor,
    TransformerSpreadPredictor,
    prepare_sequences,
    train_model,
    backtest_ou_strategy,
    PRESET_PAIRS,
)


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="StatArbEngine — OU Process + PyTorch",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #334155;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #818cf8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 4px;
    }
    .sde-box {
        background: #1e1e2e;
        border: 1px solid #444;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================
st.markdown('<p class="main-header">📈 StatArbEngine</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Ornstein-Uhlenbeck Stochastic Modeling + PyTorch Autoregressive Prediction for Pairs Trading</p>',
    unsafe_allow_html=True,
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("📊 Pair Selection")
    pair_choice = st.selectbox("Preset Pairs", list(PRESET_PAIRS.keys()))
    ticker_a, ticker_b = PRESET_PAIRS[pair_choice]

    st.text_input("Ticker A", value=ticker_a, key="ticker_a_input", disabled=True)
    st.text_input("Ticker B", value=ticker_b, key="ticker_b_input", disabled=True)

    st.subheader("📅 Date Range")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2021-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2026-09-30"))

    st.subheader("🧠 PyTorch Model")
    model_type = st.radio("Architecture", ["MLP (SpreadPredictor)", "Transformer (Self-Attention)"])
    context_length = st.slider("Context Window (days)", 5, 60, 20)
    epochs = st.slider("Training Epochs", 20, 200, 80)

    st.subheader("💹 Backtest Parameters")
    entry_z = st.slider("Entry Z-Score", 1.0, 3.5, 2.0, 0.1)
    exit_z = st.slider("Exit Z-Score", 0.0, 1.5, 0.5, 0.1)

    st.subheader("📉 Market Friction")
    transaction_bps = st.slider("Commission (bps)", 0.0, 20.0, 5.0, 1.0)
    slippage_bps = st.slider("Slippage (bps)", 0.0, 20.0, 5.0, 1.0)

    run_btn = st.button("🚀 Run Full Analysis", type="primary", use_container_width=True)


# ============================================================
# MAIN ANALYSIS
# ============================================================
if run_btn:

    # ------ STEP 1: FETCH DATA ------
    with st.spinner(f"Fetching {ticker_a} & {ticker_b} data..."):
        prices = fetch_pair_data(ticker_a, ticker_b, str(start_date), str(end_date))

    st.success(f"✅ Loaded {len(prices)} trading days of data.")

    # --- Price Plot ---
    st.header("1️⃣ Price Series & Cointegration")

    fig_prices = make_subplots(specs=[[{"secondary_y": True}]])
    fig_prices.add_trace(
        go.Scatter(x=prices.index, y=prices[ticker_a], name=ticker_a,
                   line=dict(color="#818cf8", width=2)),
        secondary_y=False,
    )
    fig_prices.add_trace(
        go.Scatter(x=prices.index, y=prices[ticker_b], name=ticker_b,
                   line=dict(color="#f472b6", width=2)),
        secondary_y=True,
    )
    fig_prices.update_layout(
        template="plotly_dark",
        title=f"{ticker_a} vs {ticker_b} — Adjusted Close",
        height=400,
        margin=dict(t=50, b=30),
    )
    st.plotly_chart(fig_prices, use_container_width=True)

    # ------ STEP 2: SPREAD & COINTEGRATION ------
    with st.spinner("Computing cointegration spread..."):
        spread_result = compute_spread(prices)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        coint_color = "🟢" if spread_result.is_cointegrated else "🔴"
        st.metric("Cointegration p-value", f"{spread_result.coint_pvalue:.4f}",
                  delta=f"{coint_color} {'Cointegrated' if spread_result.is_cointegrated else 'Not Cointegrated'}")
    with col2:
        st.metric("ADF p-value (Spread)", f"{spread_result.adf_pvalue:.4f}")
    with col3:
        st.metric("Hedge Ratio (avg)", f"{spread_result.hedge_ratio.mean():.4f}")
    with col4:
        hl = spread_result.half_life
        st.metric("Half-Life (days)", f"{hl:.1f}" if hl < 1000 else "∞")

    # --- Spread Plot ---
    fig_spread = go.Figure()
    fig_spread.add_trace(
        go.Scatter(x=spread_result.spread.index, y=spread_result.spread,
                   name="Spread", line=dict(color="#818cf8", width=1.5))
    )
    mean_val = spread_result.spread.mean()
    std_val = spread_result.spread.std()
    fig_spread.add_hline(y=mean_val, line_dash="dash", line_color="#10b981",
                         annotation_text="μ (Mean)")
    fig_spread.add_hline(y=mean_val + entry_z * std_val, line_dash="dot",
                         line_color="#ef4444", annotation_text=f"+{entry_z}σ (Short Entry)")
    fig_spread.add_hline(y=mean_val - entry_z * std_val, line_dash="dot",
                         line_color="#22c55e", annotation_text=f"-{entry_z}σ (Long Entry)")
    fig_spread.update_layout(
        template="plotly_dark", title="Cointegration Spread",
        height=350, margin=dict(t=50, b=30),
    )
    st.plotly_chart(fig_spread, use_container_width=True)

    st.divider()

    # ------ STEP 3: ORNSTEIN-UHLENBECK FIT ------
    st.header("2️⃣ Ornstein-Uhlenbeck Process — Maximum Likelihood Estimation")

    with st.spinner("Fitting OU stochastic differential equation via MLE..."):
        ou_params = fit_ou_process(spread_result.spread)

    # Display the SDE
    st.markdown("#### The Stochastic Differential Equation (SDE)")
    st.latex(r"dX(t) = \kappa \left( \mu - X(t) \right) dt + \sigma \, dW(t)")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("κ (Mean Reversion Speed)", f"{ou_params.kappa:.4f}")
    with col2:
        st.metric("μ (Equilibrium Level)", f"{ou_params.mu:.4f}")
    with col3:
        st.metric("σ (Volatility)", f"{ou_params.sigma:.4f}")
    with col4:
        st.metric("Half-Life (OU)", f"{ou_params.half_life:.1f} days")

    st.info(
        "**Physics Analogy:** This is the Langevin equation for a Brownian particle "
        f"in a harmonic potential V(x) = ½κ(x − μ)². The 'spring constant' κ = {ou_params.kappa:.4f} "
        f"pulls the spread back to equilibrium μ = {ou_params.mu:.4f} with thermal noise σ = {ou_params.sigma:.4f}.",
        icon="🔬",
    )

    st.divider()

    # ------ STEP 4: PYTORCH MODEL ------
    st.header("3️⃣ PyTorch Autoregressive Spread Prediction")

    with st.spinner("Preparing sequences & training model..."):
        train_loader, X_train, X_test, y_train, y_test, scaler = prepare_sequences(
            spread_result.spread, context_length=context_length
        )

        if "Transformer" in model_type:
            model = TransformerSpreadPredictor(context_length=context_length)
            st.info("Using **Transformer (Self-Attention)** architecture — mini-GPT for time series.", icon="🤖")
        else:
            model = SpreadPredictor(context_length=context_length)
            st.info("Using **MLP (SpreadPredictor)** architecture.", icon="🤖")

        losses = train_model(model, train_loader, epochs=epochs)

    # --- Training Loss Plot ---
    col_left, col_right = st.columns(2)
    with col_left:
        fig_loss = go.Figure()
        fig_loss.add_trace(
            go.Scatter(y=losses, mode="lines", name="Training Loss",
                       line=dict(color="#f59e0b", width=2))
        )
        fig_loss.update_layout(
            template="plotly_dark", title="Training Loss (MSE)",
            xaxis_title="Epoch", yaxis_title="Loss",
            height=350, margin=dict(t=50, b=30),
        )
        st.plotly_chart(fig_loss, use_container_width=True)

    # --- Prediction vs Actual ---
    with col_right:
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test).numpy()

        fig_pred = go.Figure()
        fig_pred.add_trace(
            go.Scatter(y=y_test.numpy().flatten(), name="Actual (Normalized)",
                       line=dict(color="#818cf8", width=1.5))
        )
        fig_pred.add_trace(
            go.Scatter(y=test_preds.flatten(), name="PyTorch Prediction",
                       line=dict(color="#ef4444", width=1.5, dash="dash"))
        )
        fig_pred.update_layout(
            template="plotly_dark",
            title="Test Set: Actual vs Predicted Spread",
            height=350, margin=dict(t=50, b=30),
        )
        st.plotly_chart(fig_pred, use_container_width=True)

    # Model stats
    total_params = sum(p.numel() for p in model.parameters())
    st.caption(f"Model: {model_type} | Parameters: {total_params:,} | Context Window: {context_length} days | Final Loss: {losses[-1]:.6f}")

    st.divider()

    # ------ STEP 5: BACKTEST ------
    st.header("4️⃣ Backtest — Pairs Trading Strategy")

    with st.spinner("Running backtest..."):
        bt = backtest_ou_strategy(
            prices, spread_result.spread, ou_params,
            entry_z=entry_z, exit_z=exit_z,
            hedge_ratios=spread_result.hedge_ratio,
            transaction_bps=transaction_bps,
            slippage_bps=slippage_bps,
        )

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        color = "normal" if bt.total_return > 0 else "inverse"
        st.metric("Total PnL", f"${bt.total_return:.2f}", delta_color=color)
    with col2:
        st.metric("Sharpe Ratio", f"{bt.sharpe_ratio:.2f}")
    with col3:
        st.metric("Max Drawdown", f"${bt.max_drawdown:.2f}")
    with col4:
        st.metric("Win Rate", f"{bt.win_rate:.1%}" if bt.num_trades > 0 else "N/A",
                  delta=f"{bt.num_trades} trades")

    # PnL Curve
    fig_pnl = go.Figure()
    fig_pnl.add_trace(
        go.Scatter(
            x=bt.pnl_curve.index, y=bt.pnl_curve.values,
            fill="tozeroy",
            line=dict(color="#10b981", width=2),
            fillcolor="rgba(16, 185, 129, 0.15)",
            name="Cumulative PnL",
        )
    )
    fig_pnl.update_layout(
        template="plotly_dark",
        title="Cumulative PnL — OU Mean-Reversion Strategy",
        xaxis_title="Date", yaxis_title="PnL ($)",
        height=400, margin=dict(t=50, b=30),
    )
    st.plotly_chart(fig_pnl, use_container_width=True)

    # Trade log
    if len(bt.trades) > 0:
        with st.expander("📋 Trade Log"):
            st.dataframe(bt.trades, use_container_width=True)

    st.divider()

    # ------ FOOTER ------
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; font-size: 0.9rem;">
            <b>StatArbEngine</b> — Ornstein-Uhlenbeck Stochastic Modeling + PyTorch Autoregressive Prediction<br/>
            Built by <b>Abhijith Krishnan B M</b> | B.Tech-M.S. Solid State Physics, IIST | Former ISRO SAC<br/>
            <i>Bridging Statistical Mechanics, Stochastic Processes, and Deep Learning for Quantitative Finance</i>
        </div>
        """,
        unsafe_allow_html=True,
    )

else:
    # Landing page
    st.markdown("""
    ### 🔬 What is this?

    **StatArbEngine** is a quantitative finance tool that combines two powerful approaches:

    1. **Classical Stochastic Modeling** — The Ornstein-Uhlenbeck (OU) process, which is the
       *exact same equation* as the Langevin equation from Statistical Mechanics. It models
       the spread between two cointegrated stocks as a Brownian particle in a harmonic potential.

    2. **PyTorch Deep Learning** — An autoregressive neural network (MLP or Transformer) trained
       to predict the next day's spread value from a context window of past values, inspired by
       Karpathy's "Zero to Hero" GPT architecture.

    ---

    #### The Physics → Finance Connection

    | Statistical Mechanics | Quantitative Finance |
    |---|---|
    | Langevin Equation: $dX = -\\gamma X \\, dt + \\sigma \\, dW$ | OU Process: $dS = \\kappa(\\mu - S) \\, dt + \\sigma \\, dW$ |
    | Brownian particle in harmonic potential | Mean-reverting spread between stock pairs |
    | Spring constant $\\gamma$ | Mean-reversion speed $\\kappa$ |
    | Thermal noise $\\sigma$ | Market volatility $\\sigma$ |
    | Relaxation time $\\tau = 1/\\gamma$ | Half-life $t_{1/2} = \\ln 2 / \\kappa$ |

    ---

    👈 **Select a pair and click "Run Full Analysis" to begin.**
    """)
