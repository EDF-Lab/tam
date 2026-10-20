# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Statistical diagnostics of a target before the fit, and of the residuals of the final model after it.

Target (training rows only): unit-root and stationarity tests (augmented Dickey-Fuller and KPSS, combined into one verdict), seasonal and trend
strength from a classical decomposition, the link between spread and level, skewness, kurtosis, Jarque-Bera, the share of outliers, the
demand class of Syntetos and Boylan, a CUSUM test for a break, and Ljung-Box. Residuals (validation rows): Ljung-Box (structure left), ARCH
(volatility clustering), Jarque-Bera, and the bias. They are hints for the user and for the search, never constraints.

The tests use the standard asymptotic critical values (MacKinnon for the Dickey-Fuller statistic, Kwiatkowski et al. for KPSS) and need
only numpy and scipy.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats

#: <diagnostics>
ADF_CRITICAL_5PCT = {"c": -2.86, "ct": -3.41}     # Dickey-Fuller, constant / constant and trend
KPSS_CRITICAL_5PCT = {"c": 0.463, "ct": 0.146}    # level / trend stationarity
OUTLIER_DEVIATIONS = 5.0
MIN_ROWS = 30


def _clean(values) -> np.ndarray:
    y = np.asarray(values, dtype=float)
    return y[np.isfinite(y)]


def adf_statistic(y: np.ndarray, trend: str = "c", lags: Optional[int] = None) -> float:
    """Augmented Dickey-Fuller t statistic of the lagged level (more negative: more evidence against a unit root)."""
    y = _clean(y)
    n = len(y)
    lags = int(round(12 * (n / 100.0) ** 0.25)) if lags is None else lags
    lags = max(0, min(lags, n // 4))
    dy = np.diff(y)
    rows = len(dy) - lags
    columns = [y[lags:-1][:rows]]
    for k in range(1, lags + 1):
        columns.append(dy[lags - k:len(dy) - k][:rows])
    columns.append(np.ones(rows))
    if trend == "ct":
        columns.append(np.arange(rows, dtype=float))
    x = np.column_stack(columns)
    z = dy[lags:][:rows]
    coef, *_ = np.linalg.lstsq(x, z, rcond=None)
    resid = z - x @ coef
    dof = max(1, rows - x.shape[1])
    sigma2 = float(resid @ resid) / dof
    covariance = sigma2 * np.linalg.pinv(x.T @ x)
    se = float(np.sqrt(max(covariance[0, 0], 1e-300)))
    return float(coef[0] / se)


def kpss_statistic(y: np.ndarray, trend: str = "c") -> float:
    """KPSS statistic (larger: more evidence against stationarity around a level or a trend)."""
    y = _clean(y)
    n = len(y)
    t = np.arange(n, dtype=float)
    design = np.column_stack([np.ones(n)] + ([t] if trend == "ct" else []))
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    e = y - design @ coef
    bandwidth = int(np.ceil(4 * (n / 100.0) ** 0.25))
    long_run = float(e @ e) / n
    for k in range(1, bandwidth + 1):
        weight = 1.0 - k / (bandwidth + 1.0)
        long_run += 2.0 * weight * float(e[k:] @ e[:-k]) / n
    return float(np.sum(np.cumsum(e) ** 2) / (n ** 2 * max(long_run, 1e-300)))


def stationarity_verdict(y: np.ndarray) -> Dict[str, Any]:
    """ADF and KPSS combined: ``stationary``, ``unit_root``, ``trend_stationary`` (a deterministic trend) or ``ambiguous``."""
    adf_c, kpss_c = adf_statistic(y, "c"), kpss_statistic(y, "c")
    adf_rejects = adf_c < ADF_CRITICAL_5PCT["c"]
    kpss_rejects = kpss_c > KPSS_CRITICAL_5PCT["c"]
    if adf_rejects and not kpss_rejects:
        verdict = "stationary"
    elif not adf_rejects and kpss_rejects:
        verdict = "unit_root"
        if adf_statistic(y, "ct") < ADF_CRITICAL_5PCT["ct"] and kpss_statistic(y, "ct") <= KPSS_CRITICAL_5PCT["ct"]:
            verdict = "trend_stationary"
    else:
        verdict = "ambiguous"
    return {"verdict": verdict, "adf": round(adf_c, 3), "kpss": round(kpss_c, 3)}


def ljung_box(x: np.ndarray, lags: int) -> Dict[str, float]:
    """Ljung-Box statistic and p-value of the autocorrelation of ``x`` up to ``lags``."""
    x = _clean(x)
    n = len(x)
    lags = max(1, min(lags, n // 3))
    x = x - x.mean()
    denominator = float(x @ x)
    if denominator == 0:
        return {"statistic": 0.0, "p_value": 1.0, "lags": lags}
    rho = np.array([float(x[k:] @ x[:-k]) / denominator for k in range(1, lags + 1)])
    q = n * (n + 2) * float(np.sum(rho ** 2 / (n - np.arange(1, lags + 1))))
    return {"statistic": q, "p_value": float(stats.chi2.sf(q, lags)), "lags": lags}


def seasonal_strength(y: np.ndarray, period: int) -> Dict[str, Optional[float]]:
    """Strength of the trend and of the seasonality (0 to 1) from a classical decomposition with a centred moving average of one period."""
    y = _clean(y)
    if period < 2 or len(y) < 2 * period + 2:
        return {"trend": None, "seasonal": None}
    kernel = np.ones(period) / period
    if period % 2 == 0:
        kernel = np.r_[0.5, np.ones(period - 1), 0.5] / period
    trend = np.convolve(y, kernel, mode="same")
    half = len(kernel) // 2
    valid = slice(half, len(y) - half)
    detrended = (y - trend)[valid]
    positions = (np.arange(len(y))[valid]) % period
    profile = np.array([detrended[positions == k].mean() if np.any(positions == k) else 0.0 for k in range(period)])
    seasonal = profile[positions]
    remainder = detrended - seasonal
    seasonal_s = max(0.0, 1.0 - float(np.var(remainder)) / max(float(np.var(remainder + seasonal)), 1e-300))
    trend_s = max(0.0, 1.0 - float(np.var(remainder)) / max(float(np.var(y[valid] - seasonal)), 1e-300))
    return {"trend": round(trend_s, 4), "seasonal": round(seasonal_s, 4)}


def demand_class(y: np.ndarray) -> Optional[str]:
    """Syntetos-Boylan class of an intermittent demand: ``smooth``, ``erratic``, ``intermittent`` or ``lumpy`` (``None`` without zeros)."""
    y = _clean(y)
    positive = y[y > 0]
    if len(positive) == 0 or len(positive) == len(y):
        return None
    adi = len(y) / len(positive)
    cv2 = float((positive.std() / positive.mean()) ** 2)
    if adi < 1.32:
        return "smooth" if cv2 < 0.49 else "erratic"
    return "intermittent" if cv2 < 0.49 else "lumpy"


def cusum_break(y: np.ndarray) -> Dict[str, Any]:
    """CUSUM test of a change of level: the largest standardised partial sum of the deviations from the mean (Kolmogorov law, 5 %)."""
    y = _clean(y)
    n = len(y)
    sd = y.std()
    if n < MIN_ROWS or sd == 0:
        return {"break": False, "position": None}
    s = np.cumsum(y - y.mean()) / (sd * np.sqrt(n))
    peak = float(np.max(np.abs(s)))
    return {"break": peak > 1.358, "position": int(np.argmax(np.abs(s))) if peak > 1.358 else None, "statistic": round(peak, 3)}


def target_diagnostics(values, period: Optional[int] = None) -> Dict[str, Any]:
    """Diagnostics of a training target (missing values allowed); ``{'n': rows}`` alone when there are too few rows.

    Args:
        values: The training values in time order.
        period: Steps of the seasonal cycle to measure (the day, the week...), ``None`` to skip the seasonal strength.
    """
    y = _clean(values)
    out: Dict[str, Any] = {"n": int(y.size)}
    if y.size < MIN_ROWS:
        return out
    median = float(np.median(y))
    mad = float(np.median(np.abs(y - median))) * 1.4826
    jb = stats.jarque_bera(y)
    out.update(
        stationarity=stationarity_verdict(y),
        skewness=round(float(stats.skew(y)), 4), kurtosis=round(float(stats.kurtosis(y)), 4), jarque_bera_p=round(float(jb.pvalue), 6),
        outlier_share=round(float(np.mean(np.abs(y - median) > OUTLIER_DEVIATIONS * mad)) if mad > 0 else 0.0, 4),
        demand_class=demand_class(y), break_=cusum_break(y),
        ljung_box_p=round(ljung_box(y, min(20, y.size // 4))["p_value"], 6),
    )
    if period:
        out["strength"] = seasonal_strength(y, int(period))
    return out


def residual_diagnostics(residuals, horizon: int = 1) -> Dict[str, Any]:
    """Diagnostics of the residuals of a model on rows it did not fit.

    Returns the Ljung-Box p-value (a small value: structure is left, add lags or an adaptive layer), the ARCH-LM p-value (volatility
    clustering: a model for the scale), Jarque-Bera, the mean (bias) and its t statistic. ``{'n': rows}`` alone when there are too few.
    """
    r = _clean(residuals)
    out: Dict[str, Any] = {"n": int(r.size)}
    if r.size < MIN_ROWS:
        return out
    lags = min(max(10, horizon), r.size // 4)
    squared = r ** 2 - np.mean(r ** 2)
    arch_p = ljung_box(squared, min(5, r.size // 6))["p_value"]
    t_bias = float(r.mean() / (r.std(ddof=1) / np.sqrt(r.size))) if r.std() > 0 else 0.0
    out.update(
        ljung_box_p=round(ljung_box(r, lags)["p_value"], 6), arch_p=round(float(arch_p), 6),
        jarque_bera_p=round(float(stats.jarque_bera(r).pvalue), 6), bias=round(float(r.mean()), 6), bias_t=round(t_bias, 3),
        structure_left=bool(ljung_box(r, lags)["p_value"] < 0.05), volatility_clustering=bool(arch_p < 0.05),
    )
    return out
#: </diagnostics>
