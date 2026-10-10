with open('README.md', 'r', encoding='utf-8') as f:
    text = f.read()

new_conclusion = '''
## Conclusions from Empirical and Synthetic Sweeps

1. **Empirical Edge (2024+ Holdout)**: Sweeping the Kalman noise ratio  = V_w/V_e$ over 7 orders of magnitude on 2018–2023 data varies the filtered spread half-life from ~11 days to under 1 day. Freezing the optimal formation tuning at ^* = 10^{-4}$ and evaluating once on 2024 onward yields a net 3 bps Sharpe between −0.99 and +0.83 across six pairs, and +0.22 on a placebo pair. With a standard error of $\\approx 0.6$, there is no statistically significant evidence of positive edge.
2. **Synthetic Validation**: The backtester was verified on planted OU spreads (50 seeds per cell). For planted half-lives of 0.9–10 days, the backtester yields a Sharpe of $\\approx$ +1.0 to +1.3 at 3 bps, compared to $\\approx$ −0.5 on a random-walk null.
3. **Kalman Gain Consistency**: As coded in erify_steady_state_gain.py, the empirical effective Kalman gain observed in the sweep matches the theoretical steady-state gain formula  = (-q_{eff} + \sqrt{q_{eff}^2 + 4q_{eff}})/2$ with a maximum absolute error of $\\approx 0.024$.

'''

start_idx = text.find('## Performance Reality')
end_idx = text.find('---', start_idx)

if start_idx != -1 and end_idx != -1:
    text = text[:start_idx] + new_conclusion + text[end_idx:]
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(text)