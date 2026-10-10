def update_readme():
    with open('README.md', 'r', encoding='utf-8') as f:
        text = f.read()

    # 1. Update Absorption point
    old_point2 = """2. **Absorption is Causally Demonstrated (Full-Pipeline Synthetic):** The V/MA sweep showed that changing $q$ didn't change profitability on *real* data. But the full-pipeline synthetic test (Section 6) shows that on a *planted* 5-day OU edge, default-q Kalman reduces oracle Sharpe from +0.97 to +0.04 (absorption), while slow-q ($10^{-5}$) recovers it to +0.82. The reason slowing $q$ doesn't help on real pairs remains unresolved: OOS half-lives are actually stable, but a high autocorrelation could just be a damped random walk lacking true mean reversion. Absorption destroys the signal at fast $q$; at slow $q$, a tradable edge simply isn't present in the real data."""
    if old_point2 in text:
        new_point2 = """2. **Absorption is Causally Demonstrated (Full-Pipeline Synthetic):** The V/MA sweep showed that changing $q$ didn't change profitability on *real* data. But the full-pipeline synthetic test (Section 6) shows that on a *planted* 5-day OU edge, default-q Kalman reduces oracle Sharpe from +0.97 to +0.02 (absorption), while slow-$q$ ($10^{-5}$) recovers it to +0.84. The reason slowing $q$ doesn't help on real pairs remains unresolved: OOS half-lives, ADF, and Variance Ratios match a ~7-day OU process, but profitability remains deeply negative, leaving it unclear if the signal simply decays or if costs (and tight thresholds) overwhelm it."""
        text = text.replace(old_point2, new_point2)
    else:
        print("Point 2 not found!")

    # 2. Add Static OLS
    old_table_end = """| **Placebo: Coke / Exxon** | 0.499 | 0.684 | 1.82 d | 12.68 d | 26 | -0.41 | -0.56 | +0.53 |"""
    if old_table_end in text:
        new_table_end = old_table_end + """\n\n### 2. Static OLS Benchmark (Formation 2018-2023, OOS 2024-Present)\nA static OLS hedge fit on the formation window and traded out-of-sample over 2024+. As shown in the synthetic simulations, a static hedge is optimal if the underlying cointegration relationship is strictly stable.\n\n| Pair | 0bps Sharpe | 3bps Sharpe | Trades |\n| :--- | :---: | :---: | :---: |\n| **Visa / Mastercard** | -0.41 | -0.57 | 26 |\n| **Coca-Cola / PepsiCo** | -0.49 | -0.59 | 25 |\n| **Goldman Sachs / Morgan Stanley** | +0.90 | +0.79 | 33 |\n| **ExxonMobil / Chevron** | -0.15 | -0.25 | 23 |\n| **JPMorgan / Bank of America** | +0.72 | +0.61 | 27 |\n| **Google / Meta** | -0.47 | -0.51 | 28 |\n| **Placebo: Coke / Exxon** | +0.50 | +0.43 | 25 |\n\nThese results echo the Kalman sweep: the maximum Sharpe is +0.79 (just over 1 SE), while most pairs—including the highly integrated Visa/Mastercard—sit solidly in the negative. With the synthetic tests showing that static OLS perfectly captures a stable structural edge, its failure to do so here provides a much stronger null result that there is simply no tradable edge in these pairs."""
        text = text.replace(old_table_end, new_table_end)
    else:
        print("Table end not found!")

    # 3. Update 'Why real pairs still earn zero'
    old_interp = """**Why real pairs still earn zero at slow $q$:** The full-pipeline synthetic at slow $q$ earns +0.84 because the *planted* OU signal is guaranteed stable over 1500 days. But as shown by the `oos_diagnostics.py` script on the real V/MA pair at $q^*$ (which uses a 120-day rolling window on out-of-sample data with correct 2018 burn-in), the OOS spread has an ADF p-value of 0.000, and VR at lags 5 and 10 of 0.82 and 0.67, which is characteristic of a true 7-day OU process. Despite this stability, a tradable edge simply isn't present in the real data (likely due to the 3 bps cost drag on standard deviations that are too tight). Absorption destroys the signal at fast $q$; at slow $q$, a profitable edge simply isn't there to recover."""
    if old_interp in text:
        new_interp = """**Why real pairs still earn zero at slow $q$ (Rigorous Null Comparison):** At $q = 10^{-4}$, a planted 5-day edge gives a Sharpe of +0.70 with an SD of $\approx 0.47$ across synthetic runs. In contrast, running a strict null (a true random-walk spread, $\phi=1$) through the identical pipeline over a matching 750-day window (`synthetic_null.py`) yields a mean Sharpe of -1.16 with a 95% bound at -0.32. The real V/MA result of -0.99 sits extremely comfortably inside the no-edge random-walk range, roughly 3.5 SDs below the expected mean for a true planted edge. \n\nFurthermore, GS/MS emerges as the best pair across multiple methods (+0.83 at $q^*$, +0.79 on Static OLS). However, they all use the exact same price path, meaning this is just a single noisy draw being measured repeatedly. A best-of-seven pick from a purely zero-edge process gives roughly +0.8 SE, perfectly matching the GS/MS result. V/MA looks far more like no edge than a planted one."""
        text = text.replace(old_interp, new_interp)
    else:
        print("Interp not found!")

    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(text)

if __name__ == "__main__":
    update_readme()
