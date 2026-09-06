# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

"""
Location-scale distributional schedule and numerics.

A 2-parameter location-scale fit is a schedule over the single P-WLS atom: fit the
location on log y, refit the scale as a Gamma GLM on the squared residuals, pick the
tail family from the residual kurtosis. StaticTAM stays a thin frontend and delegates
here; every function takes the model as first argument and builds sub-models with
type(model)(...) so this file never imports StaticTAM (no circular import).
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

_TINY: float = 1e-12
# Mean of log(chi^2_1); de-biases an L2 fit on log(r^2) back to log(sigma^2).
_LOG_CHI2_1_MEAN: float = -1.2703628454614782


# --- Standardized-residual law (Normal / Student-t) -------------------------------------------------
def select_tail_family(
    standardized_residual: np.ndarray,
    tail_family: str,
    kurtosis_threshold: float,
) -> Tuple[str, Optional[float]]:
    """Choose the standardized-residual law by excess kurtosis.

    'normal' / 'student_t' force the family; 'auto' picks Student-t only when the excess kurtosis
    exceeds kurtosis_threshold, fitting nu from it. Returns (family, nu); nu is None for the Normal.
    """
    centred = standardized_residual - standardized_residual.mean()
    variance = np.mean(centred ** 2)
    excess_kurtosis = float(np.mean(centred ** 4) / max(variance ** 2, _TINY) - 3.0)
    if tail_family == "normal":
        return "normal", None
    if tail_family == "student_t":
        return "student_t", float(np.clip(4.0 + 6.0 / max(excess_kurtosis, 1e-3), 3.0, 60.0))
    # "auto"
    if excess_kurtosis <= kurtosis_threshold:
        return "normal", None
    return "student_t", float(np.clip(4.0 + 6.0 / excess_kurtosis, 3.0, 60.0))


def standardized_ppf(tail_family: str, nu: Optional[float], tau: float) -> float:
    """Inverse CDF of the unit-variance standardized residual law at level tau."""
    if tail_family == "student_t":
        return float(stats.t.ppf(tau, nu) * np.sqrt((nu - 2.0) / nu))
    return float(stats.norm.ppf(tau))


def standardized_cdf(tail_family: str, nu: Optional[float], z: np.ndarray) -> np.ndarray:
    """CDF of the unit-variance standardized residual law at standardized value z."""
    if tail_family == "student_t":
        return stats.t.cdf(z * np.sqrt(nu / (nu - 2.0)), nu)
    return stats.norm.cdf(z)


# --- Guards / config --------------------------------------------------------------------------------
def require_distributional(model, method_name: str) -> None:
    if getattr(model, "_mode_", "plain") != "distributional":
        raise RuntimeError(
            f"{method_name}() is only available on a distributional StaticTAM "
            f"(a dict formula such as {{'mu': 'y ~ s(x)', 'sigma': '~ s(x)'}})."
        )


def init_config(model, formulas, loss, group_col, date_col, default_alpha_p,
                tail_family, kurtosis_threshold, scale_shrinkage,
                location_alpha_p, scale_alpha_p) -> None:
    """Configure a location-scale distributional container from a {param: formula} dict."""
    if "mu" not in formulas:
        raise ValueError("A distributional formula dict must contain a 'mu' (location) entry.")
    if not 0.0 <= scale_shrinkage <= 1.0:
        raise ValueError(f"scale_shrinkage must lie in [0, 1]; got {scale_shrinkage!r}")

    model.target_col_ = formulas["mu"].split("~", 1)[0].strip()
    model._mu_rhs_ = formulas["mu"].split("~", 1)[1].strip()
    model._sigma_rhs_ = (formulas["sigma"].split("~", 1)[1].strip() if "sigma" in formulas else model._mu_rhs_)

    losses = loss if isinstance(loss, dict) else {"mu": loss}
    model._location_loss_ = losses.get("mu", "l2")
    # 'gamma' -> Gamma GLM on r^2; anything else ('l2') -> L2 on log(r^2) with chi-square bias correction.
    model._scale_loss_ = "gamma" if str(losses.get("sigma", "gamma")).lower() == "gamma" else "l2"

    model.tail_family = tail_family
    model.kurtosis_threshold = float(kurtosis_threshold)
    model.scale_shrinkage = float(scale_shrinkage)
    model._location_alpha_p_ = location_alpha_p
    model._scale_alpha_p_ = scale_alpha_p if scale_alpha_p is not None else (default_alpha_p + 2.0)
    model._sub_group_col_ = group_col
    model._sub_date_col_ = date_col

    # Fitted state
    model._location_submodel_ = None
    model._scale_submodel_ = None
    model.tail_family_ = None
    model.nu_ = None
    model._global_scale_variance_ = None


# --- Helpers ----------------------------------------------------------------------------------------
def _to_model_scale(model, y: np.ndarray) -> np.ndarray:
    return np.log(np.clip(y, _TINY, None)) if model._log_target_ else np.asarray(y, dtype=float)


def _estimated(predicted: pd.DataFrame, internal_target: str) -> np.ndarray:
    return predicted[f"Estimated{internal_target}"].to_numpy(dtype=float, copy=True)


def _build_location_submodel(model, alpha_p: float):
    return type(model)(
        f"__mu__ ~ {model._mu_rhs_}", group_col=model._sub_group_col_, date_col=model._sub_date_col_,
        default_alpha_p=alpha_p, loss=model._location_loss_,
    )


def _build_scale_submodel(model):
    scale_loss = "gamma" if model._scale_loss_ == "gamma" else "l2"
    return type(model)(
        f"__sigma__ ~ {model._sigma_rhs_}", group_col=model._sub_group_col_, date_col=model._sub_date_col_,
        default_alpha_p=model._scale_alpha_p_, loss=scale_loss,
    )


def _select_location_alpha_by_cv(model, working, alpha_grid, cv_fraction) -> float:
    """Pick the location smoothing by held-out RMSE of the log target (chronological split)."""
    date_col = model._sub_date_col_
    ordered = working.sort_values(date_col) if date_col is not None else working
    split_index = int(len(ordered) * (1.0 - cv_fraction))
    split_index = min(max(split_index, 1), len(ordered) - 1)
    train_split, holdout_split = ordered.iloc[:split_index], ordered.iloc[split_index:]
    best_alpha, best_error = alpha_grid[0], np.inf
    for candidate in alpha_grid:
        sub = _build_location_submodel(model, candidate).fit(train_split)
        predicted = _estimated(sub.predict(holdout_split), "__mu__")
        error = float(np.sqrt(np.mean((holdout_split["__mu__"].to_numpy() - predicted) ** 2)))
        if error < best_error:
            best_error, best_alpha = error, candidate
    return best_alpha


# --- Schedule ---------------------------------------------------------------------------------------
def fit(model, data: pd.DataFrame, select: str = "fixed",
        cv_alpha_grid: Sequence[float] = (-6.0, -4.0, -2.0, 0.0), cv_fraction: float = 0.3):
    """Fit the location, then the scale on its squared residuals, then the tail family.

    select chooses the location smoothing: 'fixed' uses location_alpha_p; 'gcv' runs auto_fit
    (exact for the location); 'cv' picks it from cv_alpha_grid by held-out error. The scale
    smoothing stays fixed (Gaussian GCV is invalid for the Gamma GLM).
    """
    working = data.copy()
    working["__mu__"] = _to_model_scale(model, working[model.target_col_].to_numpy())

    location_alpha = model._location_alpha_p_ if model._location_alpha_p_ is not None else model.default_alpha_p_
    if select == "gcv":
        model._location_submodel_ = _build_location_submodel(model, location_alpha)
        try:
            model._location_submodel_.auto_fit(working)
        except Exception:
            model._location_submodel_.fit(working)
    elif select == "cv":
        location_alpha = _select_location_alpha_by_cv(model, working, cv_alpha_grid, cv_fraction)
        model._location_submodel_ = _build_location_submodel(model, location_alpha).fit(working)
    else:
        model._location_submodel_ = _build_location_submodel(model, location_alpha).fit(working)

    mu_hat = _estimated(model._location_submodel_.predict(working), "__mu__")
    residual = working["__mu__"].to_numpy() - mu_hat
    squared_residual = np.clip(residual ** 2, _TINY, None)
    model._global_scale_variance_ = float(np.mean(squared_residual))

    if model._scale_loss_ == "gamma":
        working["__sigma__"] = squared_residual
    else:
        working["__sigma__"] = np.log(squared_residual + _TINY)

    model._scale_submodel_ = _build_scale_submodel(model).fit(working)

    _, sigma_hat = mu_sigma(model, working)
    fit_tail_family(model, residual / sigma_hat)
    return model


def fit_tail_family(model, standardized_residual: np.ndarray) -> None:
    model.tail_family_, model.nu_ = select_tail_family(
        standardized_residual, model.tail_family, model.kurtosis_threshold
    )


def _standardized_ppf(model, tau: float) -> float:
    return standardized_ppf(model.tail_family_, model.nu_, tau)


def _standardized_cdf(model, z: np.ndarray) -> np.ndarray:
    return standardized_cdf(model.tail_family_, model.nu_, z)


def mu_sigma(model, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Return the location mu_hat (model scale) and scale sigma_hat for each row of data."""
    mu_hat = _estimated(model._location_submodel_.predict(data), "__mu__")
    scale_prediction = _estimated(model._scale_submodel_.predict(data), "__sigma__")
    if model._scale_loss_ == "gamma":
        conditional_variance = np.clip(scale_prediction, _TINY, None)
    else:
        conditional_variance = np.exp(scale_prediction - _LOG_CHI2_1_MEAN)
    if model.scale_shrinkage > 0.0 and model._global_scale_variance_ is not None:
        conditional_variance = (
            model.scale_shrinkage * model._global_scale_variance_
            + (1.0 - model.scale_shrinkage) * conditional_variance
        )
    return mu_hat, np.clip(np.sqrt(conditional_variance), _TINY, None)


# --- Prediction / scoring ---------------------------------------------------------------------------
def predict_median(model, data: pd.DataFrame) -> np.ndarray:
    require_distributional(model, "predict_median")
    mu_hat, _ = mu_sigma(model, data)
    return np.exp(mu_hat) if model._log_target_ else mu_hat


def predict_quantile(model, data: pd.DataFrame, tau) -> np.ndarray:
    require_distributional(model, "predict_quantile")
    mu_hat, sigma_hat = mu_sigma(model, data)
    tau_list = [tau] if np.isscalar(tau) else list(tau)
    columns = []
    for level in tau_list:
        model_scale_quantile = mu_hat + sigma_hat * _standardized_ppf(model, float(level))
        columns.append(np.exp(model_scale_quantile) if model._log_target_ else model_scale_quantile)
    stacked = np.stack(columns, axis=1)
    return stacked[:, 0] if np.isscalar(tau) else stacked


def predict_quantiles(model, data: pd.DataFrame, taus: Sequence[float] = (0.05, 0.5, 0.95)) -> pd.DataFrame:
    require_distributional(model, "predict_quantiles")
    levels = list(taus)
    stacked = predict_quantile(model, data, levels)
    return pd.DataFrame({f"q{level}": stacked[:, j] for j, level in enumerate(levels)}, index=data.index)


def cdf(model, data: pd.DataFrame) -> np.ndarray:
    require_distributional(model, "cdf")
    mu_hat, sigma_hat = mu_sigma(model, data)
    observed = _to_model_scale(model, data[model.target_col_].to_numpy())
    return _standardized_cdf(model, (observed - mu_hat) / sigma_hat)


def crps(model, data: pd.DataFrame, n_nodes: int = 64) -> np.ndarray:
    """CRPS per observation on the model (log) scale: closed form for Normal, else a quantile integral."""
    require_distributional(model, "crps")
    mu_hat, sigma_hat = mu_sigma(model, data)
    observed = _to_model_scale(model, data[model.target_col_].to_numpy())
    if model.tail_family_ == "normal":
        omega = (observed - mu_hat) / sigma_hat
        return sigma_hat * (
            omega * (2.0 * stats.norm.cdf(omega) - 1.0)
            + 2.0 * stats.norm.pdf(omega) - 1.0 / np.sqrt(np.pi)
        )
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    tau_grid = 0.5 * (nodes + 1.0)
    tau_weights = 0.5 * weights
    score = np.zeros_like(observed, dtype=float)
    for level, quad_weight in zip(tau_grid, tau_weights):
        quantile = mu_hat + sigma_hat * _standardized_ppf(model, float(level))
        residual = observed - quantile
        pinball = residual * (level - (residual < 0).astype(float))
        score += quad_weight * 2.0 * pinball
    return score


def anomaly_score(model, data: pd.DataFrame) -> pd.DataFrame:
    """Score each observation's abnormality against its conditional distribution."""
    require_distributional(model, "anomaly_score")
    mu_hat, sigma_hat = mu_sigma(model, data)
    observed = _to_model_scale(model, data[model.target_col_].to_numpy())
    z_score = (observed - mu_hat) / sigma_hat
    upper_tail = _standardized_cdf(model, z_score)
    tail_pvalue = np.clip(2.0 * np.minimum(upper_tail, 1.0 - upper_tail), _TINY, 1.0)
    median = np.exp(mu_hat) if model._log_target_ else mu_hat
    return pd.DataFrame({
        "predicted_median": median,
        "z_score": z_score,
        "tail_pvalue": tail_pvalue,
        "anomaly_score": -np.log(tail_pvalue),
        "side": np.where(z_score >= 0, "over", "under"),
    }, index=data.index)
