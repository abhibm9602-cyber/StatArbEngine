import re

with open('README.md', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Update Absorption point
text = re.sub(
    r'2\. \*\*Absorption is Causally Demonstrated.*?simply isn\'t present in the real data\.',
    r"2. **Absorption is Causally Demonstrated (Full-Pipeline Synthetic):** The V/MA sweep showed that changing $q$ didn't change profitability on *real* data. But the full-pipeline synthetic test (Section 6) shows that on a *planted* 5-day OU edge, default-q Kalman reduces oracle Sharpe from +0.97 to +0.02 (absorption), while slow-$q$ ($10^{-5}$) recovers it to +0.84. The reason slowing $q$ doesn't help on real pairs remains unresolved: OOS half-lives, ADF, and Variance Ratios match a ~7-day OU process, but profitability remains deeply negative, leaving it unclear if the signal simply decays or if costs (and tight thresholds) overwhelm it.",
    text, flags=re.DOTALL
)

# 2. Update Why real pairs earn zero
text = re.sub(
    r'\*\*Why real pairs still earn zero.*?simply isn\'t there to recover\.',
    r"**Why real pairs still earn zero at slow $q$ (Rigorous Null Comparison):** At $q = 10^{-4}$, a planted 5-day edge gives a Sharpe of +0.70 with an SD of $\approx 0.47$ across synthetic runs. In contrast, running a strict null (a true random-walk spread, $\phi=1$) through the identical pipeline over a matching 750-day window (`synthetic_null.py`) yields a mean Sharpe of -1.16 with a 95% bound at -0.32. The real V/MA result of -0.99 sits extremely comfortably inside the no-edge random-walk range, roughly 3.5 SDs below the expected mean for a true planted edge. \n\nFurthermore, GS/MS emerges as the best pair across multiple methods (+0.83 at $q^*$, +0.79 on Static OLS). However, they all use the exact same price path, meaning this is just a single noisy draw being measured repeatedly. A best-of-seven pick from a purely zero-edge process gives roughly +0.8 SE, perfectly matching the GS/MS result. V/MA looks far more like no edge than a planted one.",
    text, flags=re.DOTALL
)

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(text)
