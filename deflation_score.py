python

import numpy as np
from scipy.stats import entropy
def deflation_score(nominal: np.ndarray, real_proxy: np.ndarray = None, window: int = 12) -> float:
    """
    Practical Deflation Detector – 15 meaningful lines
    Returns a score in [0, 1]:
        ≤ 0.2 → real growth dominates
        0.2 – 0.4 → mild illusion possible
        ≥ 0.4 → strong warning – nominal gains are mostly inflation/deflation noise

    Based on harmonic Jensen–Shannon divergence from Frank Nielsen’s M-mixture framework
    (arXiv:1904.04017 and recent generalisations).

    Parameters
    ----------
    nominal : array-like
        Time series of values in nominal currency/units (e.g. € revenue)
    real_proxy : array-like or None
        Optional real physical quantity (e.g. kWh produced, units shipped)
        If None → uses smoothed geometric average of nominal series as proxy
    window : int
        Smoothing window for the automatic proxy (default 12 = 1 year monthly)

    Example
    -------
    >>> score = deflation_score(factory_revenue_eur)
    >>> print(f"Deflation alarm: {score:.3f}")
    """
    nominal = np.asarray(nominal, dtype=float).ravel()
    if len(nominal) < 2:
        raise ValueError("Need at least 2 points")

    # Log-growth rates of nominal series
    g_nom = np.diff(np.log(nominal))

    if real_proxy is None:
        # Automatic proxy: smoothed geometric mean of the nominal series itself
        log_smooth = np.convolve(np.log(nominal), np.ones(window)/window, mode='valid')
        real_proxy = np.exp(log_smooth)
        g_real = np.diff(np.log(real_proxy))
    else:
        real_proxy = np.asarray(real_proxy, dtype=float).ravel()
        g_real = np.diff(np.log(real_proxy[-len(nominal):]))

    # Align lengths
    min_len = min(len(g_nom), len(g_real))
    g_nom, g_real = g_nom[-min_len:], g_real[-min_len:]

    # Harmonic Jensen–Shannon divergence (closed-form, no sampling)
    eps = 1e-12
    p = np.clip(g_nom, eps, None)
    q = np.clip(g_real, eps, None)
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    js_harmonic = 0.5 * (entropy(p, m, base=2) + entropy(q, m, base=2))

    return float(np.clip(js_harmonic, 0.0, 1.0))

