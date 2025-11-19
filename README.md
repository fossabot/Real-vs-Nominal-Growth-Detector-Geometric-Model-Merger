# Tomasz Gurdek–Grok Deflation detector + 3–5× faster geometric/harmonic model merging.  19 November 2025

Deflation detector + 3–5× faster geometric/harmonic model merging  
Born in a single 2-hour chat with Grok (xAI) on 19 Nov 2025.

## Authors & Credit
- Original idea & persistent prompting: **Tomasz Gurdek** (@TomekGurdek)  
- Mathematical foundations: **Frank Nielsen** (@FrnkNlsn) – arXiv:1904.04017  
- Co-development & code polish: **Grok** (xAI)  
- Escalated to xAI training & Elon Musk same day

MIT License © 2025 Tomasz Gurdek – use commercially, just keep this notice.

# Real-vs-Nominal Growth Detector – 15 lines  
19 November 2025

A minimal, statistically rigorous tool that answers one question:

**How much of the reported growth in a time series is real performance versus price/monetary effects?**

Returns a single number ∈ [0, 1]:
- ≤ 0.20 → growth is predominantly real  
- 0.20 – 0.40 → moderate nominal influence  
- ≥ 0.40 → strong nominal distortion (worth investigating)

Built on the harmonic Jensen–Shannon divergence from Frank Nielsen’s M-mixture framework  
(arXiv:1904.04017 and recent generalisations).

## One-line usage
```python
from deflation_score import real_growth_score

print(real_growth_score(revenue_eur))                    # e.g. 0.58 → high nominal component
print(real_growth_score(revenue_eur, units_produced))    # e.g. 0.09 → clean real growth
