import streamlit as st
import numpy as np
import plotly.graph_objects as go
from engine import fetch_data, apply_kalman_filter, apply_rolling_ols, fit_ou_process_mle, backtest_vectorized, check_cointegration, check_stationarity, get_lag1_autocorr, PRESET_PAIRS

st.set_page_config(page_title="StatArbEngine", layout="wide")

with st.sidebar:
    pair_name = st.selectbox("Select Asset Pair", list(PRESET_PAIRS.keys()))
    tickers = PRESET_PAIRS[pair_name]
    baseline_ols = st.checkbox("Use Rolling OLS Baseline (Instead of Kalman)", value=False)
    entry_z = st.slider("Entry Z-Score (Train)", 1.0, 3.0, 2.0, 0.1)
    exit_z = st.slider("Exit Z-Score (Train)", 0.0, 1.0, 0.5, 0.1)
    transaction_bps = st.number_input("Transaction Cost (bps)", value=3.0)
    slippage_bps = st.number_input("Base Slippage (bps)", value=3.0)
    run_analysis = st.button("Run Full Analysis", type="primary")

if run_analysis:
    prices = fetch_data(list(tickers), "2018-01-01", "2024-01-01")
    
    train_prices = prices.iloc[:int(len(prices)*0.5)]
    coint_score, coint_pval = check_cointegration(train_prices)
    
    if baseline_ols:
        spread, hedge_ratios = apply_rolling_ols(prices)
    else:
        spread, hedge_ratios = apply_kalman_filter(prices)
        
    train_spread = spread.iloc[50:int(len(spread)*0.5)] 
    adf_pval = check_stationarity(train_spread)
    lag1_ac = get_lag1_autocorr(train_spread)
    
    st.subheader("Statistical Checks (Training Sample Only)")
    cc1, cc2, cc3 = st.columns(3)
    with cc1: st.metric("Engle-Granger Cointegration p-value", f"{coint_pval:.4f}")
    with cc2: st.metric("Spread ADF Stationarity p-value", f"{adf_pval:.4f}")
    with cc3: st.metric("Lag-1 Autocorrelation", f"{lag1_ac:.4f}")
    
    ou_params = fit_ou_process_mle(train_spread.dropna())
    half_life = 252 * np.log(2) / ou_params.kappa if ou_params.kappa > 0 else 0
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("θ (Mean Reversion Speed)", f"{ou_params.theta:.4f}")
    with c2: st.metric("μ (Equilibrium)", f"{ou_params.mu:.4f}")
    with c3: st.metric("σ (Volatility)", f"{ou_params.sigma:.4f}")
    with c4: st.metric("Raw Half-Life", f"{half_life:.2f} days")
    
    bt = backtest_vectorized(prices, spread, raw_half_life=half_life, hedge_ratios=hedge_ratios, entry_z=entry_z, exit_z=exit_z, transaction_bps=transaction_bps, base_slippage_bps=slippage_bps)
    
    st.subheader(f"Out-of-Sample Backtest ({'Rolling OLS Baseline' if baseline_ols else 'Kalman'}, Next-Open Execution)")
    st.caption("OOS Period: 2021-01-01 to 2024-01-01")
    
    bc1, bc2, bc3, bc4 = st.columns(4)
    with bc1: st.metric("OOS MTM PnL", f"${bt.total_return:.2f}")
    with bc2: st.metric("Sharpe Ratio (Annualized)", f"{bt.sharpe_ratio:.2f}")
    with bc3: st.metric("Max Drawdown", f"${bt.max_drawdown:.2f}")
    with bc4: st.metric("Total Trades", f"{bt.num_trades}")
    
    fig_pnl = go.Figure(go.Scatter(x=bt.pnl_curve.index, y=bt.pnl_curve.values, fill="tozeroy", line=dict(color="#10b981")))
    st.plotly_chart(fig_pnl, use_container_width=True)
