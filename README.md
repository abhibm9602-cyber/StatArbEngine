# StatArbEngine: Quantitative Pairs Trading Infrastructure

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![NumPy](https://img.shields.io/badge/NumPy-Optimized-green.svg)

An institutional-grade Statistical Arbitrage (StatArb) trading engine that replaces traditional static OLS regressions with a **dynamic State-Space Kalman Filter** and an **autoregressive PyTorch Causal Transformer**.

## Core Architecture

### 1. Dynamic Hedging via Kalman Filter
Traditional pairs trading relies on rolling Ordinary Least Squares (OLS), which assumes a static hedge ratio. `StatArbEngine` implements a custom NumPy vectorized Kalman Filter to recursively track the unobservable hidden state (the dynamic hedge ratio) amidst market noise.

*   **Process Noise ($Q$)**: Calibrated to $10^{-5}$ to allow the hedge ratio to drift smoothly.
*   **Measurement Noise ($R$)**: Calibrated to $10^{-3}$ to filter out high-frequency microstructure volatility.

### 2. Stochastic Modeling (Ornstein-Uhlenbeck)
The resulting spread is modeled as a mean-reverting Ornstein-Uhlenbeck (OU) stochastic process:
$$ dX_t = \theta (\mu - X_t)dt + \sigma dW_t $$
*   **Maximum Likelihood Estimation (MLE)** is used to calibrate the mean-reversion speed ($\theta$) and the volatility ($\sigma$) directly from the filtered spread.
*   Trading signals (Z-scores) are dynamically adjusted based on the OU half-life.

### 3. PyTorch Causal Transformer
A custom Self-Attention mechanism is used to predict non-linear, short-term momentum anomalies in the spread before mean-reversion occurs.
*   **Architecture**: Causal masking ensures no look-ahead bias during training.
*   **Objective**: Predict the $t+1$ directional shift of the spread.

### 4. Realistic Vectorized Backtesting
Academic backtests ignore friction. This engine incorporates:
*   **Transaction Costs**: Dynamic basis points (bps) commissions.
*   **Bid-Ask Slippage**: Volume-weighted slippage penalties.
*   **Vectorization**: Pure Pandas/NumPy execution ensuring zero `for`-loop bottlenecks across 100,000+ data points.

## Local Execution
To launch the interactive dashboard simulating the StatArb engine:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Disclaimer
*This infrastructure is for research and portfolio demonstration purposes only. It is not financial advice.*
