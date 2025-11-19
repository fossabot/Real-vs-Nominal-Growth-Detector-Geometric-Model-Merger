# Practical Deflation Detector – 15 lines

A tiny, statistically principled tool born in a 2-hour conversation with Grok (xAI) on 19 November 2025.

Given any positive-valued time series (revenue, users, tokens processed, factory output in €, …),  
`deflation_score()` returns a single number ∈ [0, 1]:

- ≤ 0.2 → real growth dominates  
- 0.2 – 0.4 → mild illusion possible  
- ≥ 0.4 → strong warning — most of the nominal gain is inflation/deflation noise

Built on the harmonic Jensen–Shannon divergence from Frank Nielsen’s M-mixture framework  
(arXiv:1904.04017 and recent generalisations).

## One-line usage
```python
from deflation_score import deflation_score

print(deflation_score(factory_revenue_eur))        # e.g. 0.58 → red flag
print(deflation_score(revenue, real_units_kwh))   # e.g. 0.09 → clean growth
