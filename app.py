import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import torch

from engine import (
    fetch_data,
    apply_kalman_filter,
    fit_ou_process_mle,
    CausalTransformer,
    prepare_transformer_data,
    train_model,
    backtest_vectorized,
    PRESET_PAIRS,
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="StatArbEngine - OU Process + PyTorch",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.image("https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python")
    st.title("⚙️ Engine Parameters")

    st.subheader("1. Data Selection")
    pair_name = st.selectbox("Select Asset Pair", list(PRESET_PAIRS.keys()))
    tickers = PRESET_PAIRS[pair_name]
    
    st.subheader("2. OU Trading Parameters")
    entry_z = st.slider("Entry Z-Score", 1.0, 3.0, 2.0, 0.1)
    exit_z = st.slider("Exit Z-Score", 0.0, 1.0, 0.5, 0.1)
    
    st.subheader("3. Friction Model")
    transaction_bps = st.number_input("Transaction Cost (bps)", value=3.0, step=0.5)
    slippage_bps = st.number_input("Base Slippage (bps)", value=3.0, step=0.5)
    st.caption("Slippage scales dynamically with market volatility.")
    
    st.subheader("4. PyTorch Architecture")
    context_length = st.slider("Context Window (Days)", 10, 60, 20)
    epochs = st.slider("Training Epochs", 20, 200, 50, 10)

    run_analysis = st.button("🚀 Run Full Analysis", use_container_width=True, type="primary")

# ============================================================
# MAIN DASHBOARD
# ============================================================
if run_analysis:
    st.title(f"📈 {tickers[0]} & {tickers[1]} Statistical Arbitrage")

    # ------ STEP 1: DATA & KALMAN FILTER ------
    st.header("1️⃣ Dynamic Hedge Ratio via Kalman Filter")
    with st.spinner("Fetching data and running State-Space Kalman Filter..."):
        prices = fetch_data(list(tickers), "2018-01-01", "2024-01-01")
        spread, hedge_ratios = apply_kalman_filter(prices)

    col1, col2 = st.columns(2)
    with col1:
        fig_prices = go.Figure()
        fig_prices.add_trace(go.Scatter(y=prices[tickers[0]], name=tickers[0]))
        fig_prices.add_trace(go.Scatter(y=prices[tickers[1]], name=tickers[1]))
        fig_prices.update_layout(template="plotly_dark", title="Raw Asset Prices")
        st.plotly_chart(fig_prices, use_container_width=True)

    with col2:
        fig_hr = go.Figure()
        fig_hr.add_trace(go.Scatter(y=hedge_ratios, name="Kalman Hedge Ratio", line=dict(color="#f59e0b")))
        fig_hr.update_layout(template="plotly_dark", title="Dynamic Hedge Ratio (State State)")
        st.plotly_chart(fig_hr, use_container_width=True)

    # Spread Visualization
    mean_val = spread.mean()
    std_val = spread.std()
    
    fig_spread = go.Figure()
    fig_spread.add_trace(go.Scatter(y=spread, name="Kalman Spread", line=dict(color="#3b82f6")))
    fig_spread.add_hline(y=mean_val, line_dash="dash", line_color="#10b981", annotation_text="μ (Mean)")
    fig_spread.add_hline(y=mean_val + entry_z * std_val, line_dash="dot", line_color="#ef4444", annotation_text=f"+{entry_z}σ (Short Entry)")
    fig_spread.add_hline(y=mean_val - entry_z * std_val, line_dash="dot", line_color="#22c55e", annotation_text=f"-{entry_z}σ (Long Entry)")
    fig_spread.update_layout(template="plotly_dark", title="Cointegration Spread (Innovation)")
    st.plotly_chart(fig_spread, use_container_width=True)

    st.divider()

    # ------ STEP 2: ORNSTEIN-UHLENBECK FIT ------
    st.header("2️⃣ Ornstein-Uhlenbeck Process — Maximum Likelihood Estimation")

    with st.spinner("Fitting OU stochastic differential equation via MLE..."):
        ou_params = fit_ou_process_mle(spread.dropna())

    st.markdown("#### The Stochastic Differential Equation (SDE)")
    st.latex(r"dX(t) = \theta \left( \mu - X(t) \right) dt + \sigma \, dW(t)")

    half_life = np.log(2) / ou_params.kappa if ou_params.kappa > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("θ (Mean Reversion Speed)", f"{ou_params.theta:.4f}")
    with c2: st.metric("μ (Equilibrium)", f"{ou_params.mu:.4f}")
    with c3: st.metric("σ (Volatility)", f"{ou_params.sigma:.4f}")
    with c4: st.metric("Half-Life (OU)", f"{half_life:.1f} days")
    st.divider()

    # ------ STEP 3: PYTORCH MODEL ------
    st.header("3️⃣ PyTorch Autoregressive Causal Transformer")

    with st.spinner("Training Causal Transformer (No Look-Ahead Bias)..."):
        train_loader, X_train, X_test, y_train, y_test, scaler = prepare_transformer_data(
            spread, context_length=context_length
        )
        model = CausalTransformer(context_length=context_length)
        losses = train_model(model, train_loader, epochs=epochs)

    cl1, cl2 = st.columns(2)
    with cl1:
        fig_loss = go.Figure(go.Scatter(y=losses, mode="lines", name="Training Loss", line=dict(color="#f59e0b")))
        fig_loss.update_layout(template="plotly_dark", title="Training Loss (MSE)")
        st.plotly_chart(fig_loss, use_container_width=True)

    with cl2:
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test).numpy()
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=y_test.numpy().flatten(), name="Actual (Normalized)", line=dict(color="#818cf8")))
        fig_pred.add_trace(go.Scatter(y=test_preds.flatten(), name="Prediction", line=dict(color="#ef4444", dash="dash")))
        fig_pred.update_layout(template="plotly_dark", title="Test Set: Actual vs Predicted")
        st.plotly_chart(fig_pred, use_container_width=True)

    st.divider()

    # ------ STEP 4: BACKTEST ------
    st.header("4️⃣ Vectorized Backtest (Daily MTM)")

    with st.spinner("Running MTM Vectorized Backtest..."):
        bt = backtest_vectorized(
            prices, spread, ou_params,
            entry_z=entry_z, exit_z=exit_z,
            hedge_ratios=hedge_ratios,
            transaction_bps=transaction_bps,
            base_slippage_bps=slippage_bps,
        )

    bc1, bc2, bc3, bc4 = st.columns(4)
    with bc1: st.metric("Total MTM PnL", f"${bt.total_return:.2f}")
    with bc2: st.metric("Sharpe Ratio (Daily)", f"{bt.sharpe_ratio:.2f}")
    with bc3: st.metric("Max Drawdown", f"${bt.max_drawdown:.2f}")
    with bc4: st.metric("Total Trades", f"{bt.num_trades}")

    fig_pnl = go.Figure(go.Scatter(x=bt.pnl_curve.index, y=bt.pnl_curve.values, fill="tozeroy", line=dict(color="#10b981")))
    fig_pnl.update_layout(template="plotly_dark", title="Cumulative MTM PnL")
    st.plotly_chart(fig_pnl, use_container_width=True)

else:
    st.markdown("### 📈 StatArbEngine\nSelect an asset pair and click **Run Full Analysis** to execute the pipeline.")
