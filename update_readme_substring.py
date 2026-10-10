with open('README.md', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace point 2
idx1 = text.find('2. **Absorption is Causally Demonstrated')
idx2 = text.find('3. **The Lookback Window Design')
if idx1 != -1 and idx2 != -1:
    new_point2 = "2. **Absorption is Causally Demonstrated (Full-Pipeline Synthetic):** The V/MA sweep showed that changing $q$ didn't change profitability on *real* data. But the full-pipeline synthetic test (Section 6) shows that on a *planted* 5-day OU edge, default-q Kalman reduces oracle Sharpe from +0.97 to +0.02 (absorption), while slow-$q$ ($10^{-5}$) recovers it to +0.84. The reason slowing $q$ doesn't help on real pairs remains unresolved: OOS half-lives, ADF, and Variance Ratios match a ~7-day OU process, but profitability remains deeply negative, leaving it unclear if the signal simply decays or if costs (and tight thresholds) overwhelm it.\n"
    text = text[:idx1] + new_point2 + text[idx2:]

# Replace interp
idx3 = text.find('**Why real pairs still earn zero at slow')
idx4 = text.find('---', idx3)
if idx3 != -1 and idx4 != -1:
    new_interp = "**Why real pairs still earn zero at slow $q$ (Rigorous Null Comparison):** At $q = 10^{-4}$, a planted 5-day edge gives a Sharpe of +0.70 with an SD of $\\approx 0.47$ across synthetic runs. In contrast, running a strict null (a true random-walk spread, $\\phi=1$) through the identical pipeline over a matching 750-day window (`synthetic_null.py`) yields a mean Sharpe of -1.16 with a 95% bound at -0.32. The real V/MA result of -0.99 sits extremely comfortably inside the no-edge random-walk range, roughly 3.5 SDs below the expected mean for a true planted edge.\n\nFurthermore, GS/MS emerges as the best pair across multiple methods (+0.83 at $q^*$, +0.79 on Static OLS). However, they all use the exact same price path, meaning this is just a single noisy draw being measured repeatedly. A best-of-seven pick from a purely zero-edge process gives roughly +0.8 SE, perfectly matching the GS/MS result. V/MA looks far more like no edge than a planted one.\n\n"
    text = text[:idx3] + new_interp + text[idx4:]

# Add Static OLS Table
table = """\n\n### 2. Static OLS Benchmark (Formation 2018-2023, OOS 2024-Present)\nA static OLS hedge fit on the formation window and traded out-of-sample over 2024+. As shown in the synthetic simulations, a static hedge is optimal if the underlying cointegration relationship is strictly stable.\n\n| Pair | 0bps Sharpe | 3bps Sharpe | Trades |\n| :--- | :---: | :---: | :---: |\n| **Visa / Mastercard** | -0.41 | -0.57 | 26 |\n| **Coca-Cola / PepsiCo** | -0.49 | -0.59 | 25 |\n| **Goldman Sachs / Morgan Stanley** | +0.90 | +0.79 | 33 |\n| **ExxonMobil / Chevron** | -0.15 | -0.25 | 23 |\n| **JPMorgan / Bank of America** | +0.72 | +0.61 | 27 |\n| **Google / Meta** | -0.47 | -0.51 | 28 |\n| **Placebo: Coke / Exxon** | +0.50 | +0.43 | 25 |\n\nThese results echo the Kalman sweep: the maximum Sharpe is +0.79 (just over 1 SE), while most pairs—including the highly integrated Visa/Mastercard—sit solidly in the negative. With the synthetic tests showing that static OLS perfectly captures a stable structural edge, its failure to do so here provides a much stronger null result that there is simply no tradable edge in these pairs.\n\n"""
idx5 = text.find('## Data Requirements')
if idx5 != -1:
    text = text[:idx5] + table + text[idx5:]

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(text)
