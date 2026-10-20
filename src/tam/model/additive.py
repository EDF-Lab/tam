# SPDX-FileCopyrightText: 2023-2026 EDF (Electricité De France)
# SPDX-FileCopyrightText: 2023-2025 Sorbonne Université
# SPDX-FileContributor: Yann Allioux
# SPDX-FileContributor: Nathan Doumèche
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Implements the core Additive TAM (StaticTAM) model.

This module defines the primary user-facing class, `StaticTAM`.
It orchestrates the assembly of effects defined in the formula, manages
data preparation, solves the primal optimization problem, and provides
methods for hyperparameter tuning and model interpretation.
"""

from typing import Dict, List, Any, Tuple, Optional, Union, Sequence
import re
import warnings
import torch
import pandas as pd
import numpy as np 

from tam.common.utils import (
    TORCH_DEVICE, _check_features, _balance_groups, parse_formula_to_terms, 
    _ensure_dummies, _cleanup_dummies
)
from tam.common.exceptions import warn_extrapolation, EXTRAPOLATION_TOLERANCE
from ._base import BaseTAM
from .safety import SafetyTAM
from ._data import (
    _resolve_fixed_ranges,
    _fit_normalization_params,
    _transform_data_stacked,
    _groups_in_data,
    _check_known_groups,
    _coefficients_of_groups,
    _reassemble_decomposed_predictions
)

from ._dispatcher import smart_solve, smart_evaluate_rmse, smart_decompose
from ._dispatcher_gcv import smart_solve_gcv

from .spectrum import (
    BaseEffect, 
    OffsetEffect, LinearEffect, FourierEffect, SplineEffect,
    CategoricalEffect, ChebyshevEffect, WaveletEffect,
    NeuralEffect, RBFEffect, UniversalPhysicsEffect,
    TensorProductEffect, TreeEffect, LinearTreeEffect,
    create_effects_from_parsed_terms,
    initialize_effects,
    categorical_ranges,
    resolve_vocabulary,
    fixed_ranges,
    categorical_features,
    extrapolating_features,
    build_phi_from_effects,
    build_penalty_from_effects
)
from .statistics.estimation import build_strategy, _distributional, _mixture
from .statistics.estimation._reweighting import _linear_predictor
from .statistics.estimation import _target_transform as _tt
from .statistics.estimation._count_families import predict_count_quantiles

_TINY: float = 1e-12
#: Reweighting passes of the working-weight GCV of a GLM loss, and the move (in log10 of lambda) under which they stop.
_GCV_REWEIGHTING_PASSES: int = 4
_GCV_REWEIGHTING_TOL: float = 0.25

class StaticTAM(BaseTAM):
    """
    Implements the formula-driven Additive TAM model.

    This class builds interpretable, physics-informed models by summing
    various functional effects (Linear, Splines, Fourier, Neural, etc.) defined
    via an R-style formula.

    The model minimizes the penalized empirical risk by solving the regularized
    normal equations in the primal space. See the theoretical documentation for
    exact mathematical proofs.

    Attributes:
        effects_list_ (List[BaseEffect]): The list of instantiated effect objects.
        coefficients_ (torch.Tensor): The fitted coefficients (batch, n_coeffs, 1).
        norm_params_ (dict): Parameters used for data normalization.
    """
    
#: <init_additive>
    def __init__(
        self,
        formula: Union[str, Dict[str, str]] = None,
        group_col: str = None,
        date_col: str = None,
        default_alpha_p: float = -9.0,
        loss: Union[str, Dict[str, str]] = "l2",
        loss_kwargs: Optional[dict] = None,
        dist_kwargs: Optional[dict] = None,
        mixture_components: Optional[int] = None,
        mixture_kwargs: Optional[dict] = None,
        _internal_effects_list: Optional[List[BaseEffect]] = None,
        _internal_features_config: Optional[dict] = None,
        target_transform: Optional[str] = None,
        transform_kwargs: Optional[dict] = None
    ):
        """
        Initializes the StaticTAM model.

        Args:
            formula: R-style formula defining the model structure
                     (e.g., "Y ~ s(x) + l(t)").
            group_col: Column name used for grouping data (e.g., 'ID').
            date_col: Column name for time indexing. Defaults to None, in which
                case rows are tensorized in whatever order they were passed in.
                For formulas with temporal terms (lags, trend, or any term whose
                penalty assumes row-adjacency means chronological adjacency),
                you must pass date_col explicitly to guarantee correct results
                on data that isn't already sorted by time.
            default_alpha_p: Default log10(lambda_p) regularization strength.
            dist_kwargs: Options of the distributional and mixture modes: ``tail_family``, ``kurtosis_threshold``,
                ``scale_shrinkage``, ``location_alpha_p``, ``scale_alpha_p``, and the target scale: ``target_transform``
                (``"log"`` by default, ``"none"``, ``"asinh"`` for a skewed target with negative values, ``"logit"`` for a
                bounded one, with ``bounds=(low, high)`` or estimated) or the older ``log_target`` (True = ``"log"``).
            target_transform: Fit on ``g(y)`` and return the response scale: ``"log"``, ``"log1p"``, ``"asinh"`` (a skewed target with
                negative values), ``"logit"`` (a bounded one) or ``"boxcox"``. ``predict`` inverts the fitted value, ``predict_quantiles``
                inverts the quantiles of the model-scale residuals (exact for a monotone ``g``). Not available with a GLM loss (its link
                already is a transform), a dict formula or a mixture: those take ``dist_kwargs``.
            transform_kwargs: Parameters of the transform: ``shift`` (``log``, ``boxcox``), ``lam`` (``boxcox``, else maximum likelihood),
                ``centre`` and ``scale`` (``asinh``), ``bounds`` (``logit``), and ``mean_correction=True`` for a mean forecast
                (second-order correction ``E[y] = g^-1(z) + (g^-1)''(z) var / 2``) instead of the median-type back-transform.
            _internal_effects_list: (Internal) Used for restoring state during grid search.
            _internal_features_config: (Internal) Used for restoring state during grid search.
        """
        super().__init__()

        if target_transform is not None:
            if target_transform not in _tt.TRANSFORMS:
                raise ValueError(f"target_transform must be one of {_tt.TRANSFORMS}; got {target_transform!r}")
            if isinstance(formula, dict) or mixture_components is not None:
                raise ValueError("target_transform applies to a plain StaticTAM; a distributional or mixture model takes dist_kwargs={'target_transform': ...}.")
            _tt.check_kwargs(target_transform, dict(transform_kwargs or {}))
        self._offset_features_ = []
        if isinstance(formula, str) and "~" in formula:
            _, all_terms = parse_formula_to_terms(formula)
            self._offset_features_ = [term['feature'] for term in all_terms if term['type'] == 'offset']
            if self._offset_features_ and mixture_components is not None:
                raise ValueError("offset(...) is not available in a mixture model.")
        self._plain_transform_ = "none" if target_transform is None else target_transform
        self._plain_kwargs_ = dict(transform_kwargs or {})
        self._plain_params_ = {}
        self._plain_variance_ = 0.0
        self._plain_residual_levels_ = None
        self._plain_residual_values_ = None
        
        self.effects_list_ = []
        self.formula_ = formula 
        self.default_alpha_p_ = default_alpha_p
        self.group_col_ = group_col or "__dummy_group__"
        self.date_col_ = date_col or "__dummy_date__"
        self.is_grid_search_template_ = False

        dk = dist_kwargs or {}
        # Extracted universally since both distributional and mixture modes use the target scale transformation.
        self._target_transform_, self._log_target_, self._transform_bounds_ = _distributional.resolve_target_transform(
            dk, mixture=mixture_components is not None
        )
        # Conformal state (set by calibrate_conformal; used by predict_intervals).
        self._safety_ = None
        self._conformal_studentized_ = False

        # Distributional mode: a dict formula makes StaticTAM a thin location-scale container;
        # the schedule and its predictions live in _distributional.py (sub-models hold the effects).
        if isinstance(formula, dict):
            if mixture_components is not None:
                raise ValueError("mixture_components is incompatible with a distributional (dict) formula.")
            self._mode_ = "distributional"
            _distributional.init_config(
                self, formula, loss, group_col, date_col, default_alpha_p,
                tail_family=dk.get("tail_family", "auto"),
                kurtosis_threshold=dk.get("kurtosis_threshold", 1.0),
                scale_shrinkage=dk.get("scale_shrinkage", 0.0),
                location_alpha_p=dk.get("location_alpha_p"),
                scale_alpha_p=dk.get("scale_alpha_p"),
            )
            return
        self._mode_ = "plain"

        # Mixture (mixture_components=K): EM whose M-step is the weighted atom; schedule in _mixture.py.
        mk = mixture_kwargs or {}
        self._mixture_components_ = int(mixture_components) if mixture_components is not None else None
        self._mixture_seed_ = int(mk.get("seed", 0))
        self._mixture_n_init_ = max(int(mk.get("n_init", 1)), 1)
        self._mixture_max_iter_ = int(mk.get("max_iter", 100))
        self._mixture_tol_ = float(mk.get("tol", 1e-5))
        self.component_coefficients_ = []
        self.component_scales_ = np.zeros(0)
        self.mixing_weights_ = np.zeros(0)
        self.log_likelihood_ = -np.inf

        # scalar-loss reweighting strategy. None (l2/gaussian/normal, or mixture mode) keeps the default
        # single unweighted solve; any other loss builds an IRLS strategy from loss_kwargs.
        self.loss_ = loss
        self._loss_kwargs_ = dict(loss_kwargs or {})
        self.dispersion_ = None
        if self._plain_transform_ != "none" and str(loss).lower() not in ("l2", "gaussian", "normal", "expectile", "huber", "student_t"):
            raise ValueError(f"target_transform={self._plain_transform_!r} cannot be combined with loss={loss!r}: the link of that family is already a transform.")
        if self._mixture_components_ is not None:
            self._reweighting_strategy_ = None
        else:
            self._reweighting_strategy_ = (
                None if str(loss).lower() in ("l2", "gaussian", "normal")
                else build_strategy(loss, **(loss_kwargs or {}))
            )

        if _internal_effects_list:
            self.effects_list_ = _internal_effects_list
            self.features_config_ = _internal_features_config
            self.is_grid_search_template_ = False

        elif formula:
            self.target_col_, self.parsed_terms_ = parse_formula_to_terms(formula)
            real_features = self._extract_recursive_features(self.parsed_terms_)
            if self._offset_features_:
                # An offset is read from the raw column: no effect, no coefficient, and no other term may read the same column.
                self.parsed_terms_ = [term for term in self.parsed_terms_ if term['type'] != 'offset']
                # The solvers route the feature columns to the effects by position: the offsets go last, after every column an effect reads.
                real_features = [name for name in real_features if name not in self._offset_features_] + list(self._offset_features_)
                shared = sorted(set(self._offset_features_) & set(self._extract_recursive_features(self.parsed_terms_)))
                if shared:
                    raise ValueError(f"offset({shared[0]}) reads the raw column; another term of the formula also reads '{shared[0]}'. "
                                     "Give the offset its own column.")
            self.features_config_ = { "features": real_features }
            
            self.is_grid_search_template_ = False
            
            try:
                # Attempt standard instantiation; valid string hyperparams will pass.
                self.effects_list_ = create_effects_from_parsed_terms(
                    self.parsed_terms_, 
                    token_values={}, 
                    default_alpha_p=self.default_alpha_p_
                )
            except Exception as e:
                # Missing categorical counts are filled later in _prepare_data
                if isinstance(e, ValueError) and "requires 'n_cat'" in str(e):
                    pass
                else:
                    # If instantiation structurally fails and strings are present, it implies grid tokens are actively blocking the types.
                    has_str_vals = any(isinstance(v, str) for t in self.parsed_terms_ for v in t['params'].values())
                    if has_str_vals:
                        self.is_grid_search_template_ = True
                        print("Model initialized with Grid Search tokens. Use 'grid_search_fit()'.")
                    else:
                        raise e
        else:
            raise ValueError("`formula` must be provided to initialize StaticTAM.")
#: </init_additive>

    def fit(self, data_train: pd.DataFrame, **schedule_kwargs) -> "StaticTAM":
        """Fit, routing to the schedule implied by the constructor inputs.

        Distributional (dict formula) -> distributional; mixture (mixture_components) -> EM; otherwise the
        standard BaseTAM.fit (one exact solve for l2, else IRLS).
        """
        if getattr(self, "_mode_", "plain") == "distributional":
            return _distributional.fit(self, data_train, **schedule_kwargs)
        if getattr(self, "_mixture_components_", None) is not None:
            return _mixture.fit(self, data_train)
        if self._transform_active():
            working = self._transformed_frame(data_train, estimate=True)
            super().fit(working)
            self._finish_transform_fit(working)
            return self
        return super().fit(data_train)

    # --- Target transform of the plain mode (statistics.estimation._target_transform) ---
    def _transform_active(self) -> bool:
        return getattr(self, "_plain_transform_", "none") != "none"

    def _transformed_frame(self, data: pd.DataFrame, estimate: bool = False) -> pd.DataFrame:
        """A copy of ``data`` whose target is ``g(y)``; ``estimate`` fits the parameters of ``g`` on it first."""
        if not self._transform_active():
            return data
        values = data[self.target_col_].to_numpy(dtype=float)
        if estimate:
            self._plain_params_ = _tt.estimate_params(self._plain_transform_, values, self._plain_kwargs_)
        working = data.copy()
        working[self.target_col_] = _tt.forward(self._plain_transform_, self._plain_params_, values)
        return working

    def _finish_transform_fit(self, working: pd.DataFrame) -> None:
        """Store the law of the model-scale residuals (variance and quantile function) that ``predict_quantiles`` and the mean correction use."""
        fitted = BaseTAM.predict(self, working)[f"Estimated{self.target_col_}"].to_numpy(dtype=float)
        residual = working[self.target_col_].to_numpy(dtype=float) - fitted
        residual = residual[np.isfinite(residual)]
        if residual.size == 0:
            residual = np.zeros(1)
        self._plain_variance_ = float(np.var(residual))
        self._plain_residual_levels_ = np.linspace(0.0, 1.0, int(min(max(residual.size, 2), 20001)))
        self._plain_residual_values_ = np.quantile(residual, self._plain_residual_levels_)

    def _model_scale_prediction(self, data: pd.DataFrame) -> np.ndarray:
        return BaseTAM.predict(self, data)[f"Estimated{self.target_col_}"].to_numpy(dtype=float, copy=True)

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict, mapping GLM losses back to the mean scale via the inverse link.

        For loss='l2' and the identity-link M-estimators (expectile / huber / student_t) the base
        prediction is returned unchanged. For GLM losses (gamma / poisson / binomial) the estimate is
        mapped from the link scale to the mean scale: mu = g^-1(eta).
        """
        if getattr(self, "_mode_", "plain") == "distributional":
            raise RuntimeError(
                "This is a distributional StaticTAM; use predict_median(), predict_quantiles(), "
                "cdf() or anomaly_score() instead of predict()."
            )
        if getattr(self, "_mixture_components_", None) is not None:
            raise RuntimeError(
                "This is a mixture StaticTAM; use predict_mean(), component_means(), "
                "responsibilities() or anomaly_score() instead of predict()."
            )
        predicted = super().predict(data)
        strategy = getattr(self, "_reweighting_strategy_", None)
        if strategy is not None and getattr(strategy, "is_glm", False):
            estimated_column = f"Estimated{self.target_col_}"
            if estimated_column in predicted.columns:
                linear_predictor = torch.as_tensor(
                    predicted[estimated_column].to_numpy(copy=True), dtype=torch.get_default_dtype()
                )
                predicted[estimated_column] = strategy.inverse_link(linear_predictor).cpu().numpy()
        if self._transform_active():
            estimated_column = f"Estimated{self.target_col_}"
            if estimated_column in predicted.columns:
                z_hat = predicted[estimated_column].to_numpy(dtype=float, copy=True)
                if self._plain_kwargs_.get("mean_correction", False):
                    predicted[estimated_column] = _tt.inverse_mean(self._plain_transform_, self._plain_params_, z_hat, self._plain_variance_)
                else:
                    predicted[estimated_column] = _tt.inverse(self._plain_transform_, self._plain_params_, z_hat)
        return predicted

    # --- Distributional API: thin delegates to statistics.estimation._distributional ---
    def _to_model_scale(self, y: np.ndarray) -> np.ndarray:
        return _distributional._to_model_scale(self, y)

    def _from_model_scale(self, z: np.ndarray, quantity: str = "the back-transformed values") -> np.ndarray:
        return _distributional.from_model_scale(self, z, quantity)

    def _mu_sigma(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        return _distributional.mu_sigma(self, data)

    def predict_median(self, data: pd.DataFrame) -> np.ndarray:
        return _distributional.predict_median(self, data)

    def predict_quantile(self, data: pd.DataFrame, tau: Union[float, Sequence[float]]) -> np.ndarray:
        return _distributional.predict_quantile(self, data, tau)

    def predict_quantiles(self, data: pd.DataFrame, taus: Sequence[float] = (0.05, 0.5, 0.95)) -> pd.DataFrame:
        """Quantiles on the response scale: of the location-scale law (distributional mode), or of the fitted law of a
        ``poisson``, ``negative_binomial`` or ``tweedie`` loss (discrete quantiles for the counts, the compound Poisson-gamma
        law for Tweedie; non-negative by construction)."""
        strategy = getattr(self, "_reweighting_strategy_", None)
        if getattr(self, "_mode_", "plain") != "distributional" and hasattr(strategy, "quantiles"):
            return predict_count_quantiles(self, data, taus)
        if self._transform_active():
            levels = list(taus)
            offsets = np.interp(levels, self._plain_residual_levels_, self._plain_residual_values_)
            z_hat = self._model_scale_prediction(data)
            quantiles = {f"q{level}": _tt.inverse(self._plain_transform_, self._plain_params_, z_hat + offset)
                         for level, offset in zip(levels, offsets)}
            return pd.DataFrame(quantiles, index=data.index)
        return _distributional.predict_quantiles(self, data, taus)

    def cdf(self, data: pd.DataFrame) -> np.ndarray:
        return _distributional.cdf(self, data)

    def crps(self, data: pd.DataFrame, n_nodes: int = 64) -> np.ndarray:
        return _distributional.crps(self, data, n_nodes)

    def anomaly_score(self, data: pd.DataFrame):
        """Distributional mode: a frame (z_score, tail_pvalue, ...). Mixture mode: the -log_density array."""
        if getattr(self, "_mode_", "plain") == "distributional":
            return _distributional.anomaly_score(self, data)
        if getattr(self, "_mixture_components_", None) is not None:
            return _mixture.anomaly_score(self, data)
        raise RuntimeError("anomaly_score() requires a distributional (dict-formula) or mixture StaticTAM.")

    # --- Mixture API: thin delegates to statistics.estimation._mixture ---
    def predict_mean(self, data: pd.DataFrame) -> np.ndarray:
        return _mixture.predict_mean(self, data)

    def component_means(self, data: pd.DataFrame) -> np.ndarray:
        return _mixture.component_means(self, data)

    def responsibilities(self, data: pd.DataFrame) -> np.ndarray:
        return _mixture.responsibilities(self, data)

    def log_density(self, data: pd.DataFrame) -> np.ndarray:
        return _mixture.log_density(self, data)

    # --- Conformal wrappers: thin delegates to SafetyTAM (static) / risk.aci (streaming). No conformal
    #     math here, this class only picks a point estimate / scale from its own predictions. ---
    def _point_prediction(self, data: pd.DataFrame) -> np.ndarray:
        """A single point-estimate array for conformal calibration, per mode."""
        if self._mode_ == "distributional":
            return self.predict_median(data)
        if self._mixture_components_ is not None:
            return self.predict_mean(data)
        return self.predict(data)[f"Estimated{self.target_col_}"].to_numpy(dtype=float, copy=True)

    def _conformal_scale(self, data: pd.DataFrame) -> np.ndarray:
        """Response-scale one-sigma half-width (Q0.841 - Q0.159)/2 used to studentize conformal scores."""
        lower = self.predict_quantile(data, 0.15865)
        upper = self.predict_quantile(data, 0.84135)
        return np.clip((upper - lower) / 2.0, _TINY, None)

    def calibrate_conformal(self, calibration_data: pd.DataFrame, alpha: float = 0.1,
                            studentized: bool = False) -> "StaticTAM":
        """Calibrate a distribution-free conformal engine on a hold-out set (delegates to SafetyTAM)."""
        y_true = calibration_data[self.target_col_].to_numpy()
        y_pred = self._point_prediction(calibration_data)
        self._conformal_studentized_ = bool(studentized and self._mode_ == "distributional")
        scale = self._conformal_scale(calibration_data) if self._conformal_studentized_ else None
        self._safety_ = SafetyTAM(alpha).calibrate(y_true, y_pred, scale=scale)
        return self

    def predict_intervals(self, data: pd.DataFrame, method: str = "static", gamma: float = 0.05,
                          y_true_online: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Conformal prediction intervals around the point estimate (requires calibrate_conformal first)."""
        if self._safety_ is None:
            raise RuntimeError("Call calibrate_conformal() before predict_intervals().")
        y_pred = self._point_prediction(data)
        scale = self._conformal_scale(data) if getattr(self, "_conformal_studentized_", False) else None
        if method == "static":
            return self._safety_.predict_intervals(y_pred, y_true_online=y_true_online, scale=scale)
        if method == "aci":
            from .statistics.risk.aci import adaptive_conformal_intervals
            return adaptive_conformal_intervals(
                self._safety_.conformal_quantile, y_pred, y_true_online,
                self._safety_.alpha_target, gamma=gamma, scale=scale,
            )
        raise ValueError("method must be 'static' or 'aci'.")

    def _extract_recursive_features(self, terms: List[Dict]) -> List[str]:
        """
        Recursively extracts physical feature names, preserving the order of appearance.
        Dives into Tensor Products and parses 'others' arguments.
        """
        features_ordered = []
        seen = set()

        def add_feature(f_name: str):
            if f_name not in seen:
                seen.add(f_name)
                features_ordered.append(f_name)

        for term in terms:
            if term['type'] != 'te':
                add_feature(term['feature'])
            
            if 'others' in term['params'] and isinstance(term['params']['others'], str):
                others_str = term['params']['others']
                extras = [s.strip() for s in others_str.split('|') if s.strip()]
                for extra in extras:
                    add_feature(extra)

            if 'slope' in term['params'] and isinstance(term['params']['slope'], str):
                add_feature(term['params']['slope'].strip())

            if term['type'] == 'te':
                for sub_term_raw in term['params'].keys():
                    if not sub_term_raw: continue
                    try:
                        _, sub_parsed = parse_formula_to_terms(f"DUMMY ~ {sub_term_raw}")
                        sub_features = self._extract_recursive_features(sub_parsed)
                        for sub_f in sub_features:
                            add_feature(sub_f)
                    except Exception:
                        pass
        
        return features_ordered
    
    def _get_data_info(self, data: pd.DataFrame, with_levels: bool = False) -> Dict[str, Any]:
        """Safely infers the number of categories (max + 1) for features.

        With ``with_levels`` the integer-coded features also get ``'__levels__'``: the number of distinct codes, which a ``c()`` term
        without ``n_cat`` takes as its level count (see ``resolve_vocabulary``).
        """
        info: Dict[str, Any] = {}
        levels: Dict[str, int] = {}
        for col in self.features_config_.get('features', []):
            if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                info[col] = int(data[col].max(skipna=True)) + 1
                codes = data[col].dropna()
                if with_levels and len(codes) and bool((codes == np.round(codes)).all()):
                    levels[col] = int(codes.nunique())
        if with_levels:
            info['__levels__'] = levels
        return info

    def _offset_of(self, x_data: torch.Tensor) -> Optional[torch.Tensor]:
        """The sum of the ``offset(col)`` columns of the formula, read from the (identity-normalised) feature tensor."""
        names = getattr(self, "_offset_features_", None)
        if not names:
            return None
        features = self.features_config_['features']
        columns = [features.index(name) for name in names]
        return x_data[..., columns].sum(dim=-1, keepdim=True)

    def _apply_vocabulary(self, data: pd.DataFrame) -> pd.DataFrame:
        """Replaces the codes of the vocabulary features by their level index; a code not in the training vocabulary gets the index
        past the last level, where the categorical effect is zero. NaN stays NaN."""
        vocabulary = getattr(self, "_vocabulary_", None)
        if not vocabulary:
            return data
        mapped = data.copy()
        for name, codes in vocabulary.items():
            if name not in mapped.columns:
                continue
            values = mapped[name].to_numpy(dtype=float)
            position = np.clip(np.searchsorted(codes, values), 0, len(codes) - 1)
            known = codes[position] == values
            indices = np.where(known, position, len(codes)).astype(float)
            mapped[name] = np.where(np.isnan(values), np.nan, indices)
        return mapped
    
    def _warn_missing_levels(self, data: pd.DataFrame) -> None:
        r"""
        Training-time information: a ``c()`` term whose ``n_cat`` is given in the formula expects the levels ``0 .. n_cat - 1``
        (``1 .. n_cat`` when the codes do not fit there); the levels the training rows do not hold have no data behind them.
        Features with non-integer codes (a coordinate read by a Fourier topology) are skipped.
        """
        for effect in self.effects_list_:
            if not isinstance(effect, CategoricalEffect) or not getattr(effect, "n_cat_given", True):
                continue
            if effect.feature_name not in data.columns:
                continue
            codes = data[effect.feature_name].dropna()
            if codes.empty or not (codes == np.round(codes)).all():
                continue
            n = effect.n_categories
            start = 0 if codes.max() <= n - 1 else int(codes.min())
            missing = sorted(set(range(start, start + n)) - set(codes.astype(int).unique().tolist()))
            if not missing:
                continue
            effect_of_missing = {
                "nominal": "Their effect is regularized to 0.",
                "ordinal": "Their effect follows from their neighbours through the smoothness penalty.",
                "fourier": "They have no estimate of their own (the effect is a smooth periodic curve).",
            }[effect.topology]
            warnings.warn(
                f"TAM [Info]: the categorical feature '{effect.feature_name}' is configured with {n} categories, but "
                f"{len(missing)} are missing from the training data: {missing[:12]}{'...' if len(missing) > 12 else ''}. {effect_of_missing}",
                UserWarning, stacklevel=4)

    def summary(self) -> pd.DataFrame:
        """
        Generates a structured summary of the model architecture.
        
        Returns:
            pd.DataFrame: A table describing each effect, its complexity (degrees of freedom),
                          structural parameters (e.g., knots, PDE weights), and regularization strength.
        """
        if not self.effects_list_:
            return pd.DataFrame()
            
        summary_data = []
        
        for effect in self.effects_list_:
            eff_type_raw = effect.__class__.__name__.replace('Effect', '')
            name = effect.feature_name
            complexity = effect.get_n_coeffs()
            details = ""
            
            if isinstance(effect, OffsetEffect):
                details = "Global Bias"
            elif isinstance(effect, LinearEffect):
                details = "Slope (Ridge)"
            elif isinstance(effect, FourierEffect):
                details = f"m={effect.m}, Sobolev(s={effect.s})"
            elif isinstance(effect, SplineEffect) and not isinstance(effect, UniversalPhysicsEffect):
                details = f"Knots={effect.n_knots}, Diff(d={effect.penalty_order})"
            elif isinstance(effect, CategoricalEffect):
                details = f"Cats={effect.n_categories}, Topo={effect.topology}"
            elif isinstance(effect, NeuralEffect):
                details = f"Neurons={effect.n_neurons}, Act={effect.activation}"
            elif isinstance(effect, UniversalPhysicsEffect):
                ops = [f"{k}={v}" for k, v in effect.diff_weights.items()]
                details = f"Basis={effect.basis_type}, PDE[{', '.join(ops)}]"
                eff_type_raw = "Physics"
            elif isinstance(effect, WaveletEffect):
                details = f"Scales={effect.n_scales}, Locs={effect.n_locations}"
            elif isinstance(effect, RBFEffect):
                details = f"Centers={effect.n_centers}" + (f", Gamma={effect.gamma:.2f}" if effect.gamma else "")
            elif isinstance(effect, ChebyshevEffect):
                details = f"Deg={effect.degree}, Sobolev(s={effect.s})"
            elif isinstance(effect, TensorProductEffect):
                details = f"Interaction ({len(effect.effects)} terms)"
            elif isinstance(effect, TreeEffect):
                details = f"Trees={effect.n_trees}, MaxDepth={effect.max_depth}"
            else:
                details = "Custom"

            coordinates = effect.penalty_coordinates()
            logs = [round(float(np.log10(c)), 2) if c > 0 else -np.inf for c in coordinates]
            lambda_p_log = logs[0] if len(logs) == 1 else " / ".join(f"{v:.2f}" for v in logs)

            summary_data.append({
                "Feature": name,
                "Type": eff_type_raw,
                "Complexity (D)": complexity,
                "Structure / Params": details,
                "Reg (log10)": lambda_p_log
            })

        for column in getattr(self, "_offset_features_", []):
            summary_data.append({
                "Feature": column,
                "Type": "ExposureOffset",
                "Complexity (D)": 0,
                "Structure / Params": "Fixed (coeff=1)",
                "Reg (log10)": "-"
            })
            
        return pd.DataFrame(summary_data)

    def _prepare_data(
        self, 
        data: pd.DataFrame, 
        target_col: Optional[str] = None,
        ignore_template_check: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, List]:
        """Prepares data for the additive model (normalization and stacking)."""
        
        if self.coefficients_ is None and self.is_grid_search_template_ and not ignore_template_check:
            raise RuntimeError("Model has grid tokens. Call 'grid_search_fit()' first.")
                             
        current_target_col = target_col if target_col is not None else self.target_col_

        if not self.effects_list_ and not self.is_grid_search_template_:
            data_info = self._get_data_info(data, with_levels=True)
            self.effects_list_ = create_effects_from_parsed_terms(
                self.parsed_terms_,
                token_values={},
                default_alpha_p=self.default_alpha_p_,
                data_info=data_info
            )
            self._vocabulary_ = {
                name: np.unique(data[name].dropna().to_numpy(dtype=float))
                for name in resolve_vocabulary(self.effects_list_, data_info)
            }

        mapped_data = self._apply_vocabulary(data)
        if self.norm_params_ is None:
            self.norm_params_, self.unique_groups_ = _fit_normalization_params(
                data=mapped_data,  
                features=self.features_config_["features"], 
                group_col=self.group_col_,
                categorical_levels=categorical_ranges(self.effects_list_),
                fixed_ranges={**_resolve_fixed_ranges(data, fixed_ranges(self.effects_list_)),
                              **{name: (-1.0, 1.0) for name in getattr(self, "_offset_features_", [])}}
            )
            
        if target_col is not None:
            self._warn_missing_levels(data)

        x_stacked, y_stacked = _transform_data_stacked(
            data=mapped_data,
            features=self.features_config_["features"],
            group_col=self.group_col_,
            norm_params=self.norm_params_,
            target_col=current_target_col,  
            unique_groups=self.unique_groups_,
            date_col=self.date_col_
        )

        if torch.isnan(x_stacked).any():
            raise ValueError(
                "TAM [Data Error]: The input features (X) contain NaN values. "
                "Please clean or impute your dataset."
            )
            
        # Trees and RBF centres are set from the full training tensor, never from a memory probe or a chunk.
        # Only the training call reaches this with uninitialised effects; later calls leave them unchanged.
        if target_col is not None:
            feature_names = self.features_config_['features'] if self.features_config_ else None
            initialize_effects(x_stacked, self.effects_list_, feature_columns=feature_names)

        if target_col is not None and y_stacked is not None:
            if torch.isnan(y_stacked).any():
                raise ValueError(
                    f"TAM [Data Error]: The target column '{target_col}' "
                    "contains NaN values. Cannot proceed with optimization."
                )
        if target_col is not None:
            self._seen_levels_ = {
                name: set(np.unique(data[name].dropna().to_numpy()).tolist())
                for name in categorical_features(self.effects_list_) if name in data.columns
            }
        else:
            self._warn_extrapolation(data, x_stacked)

        # The groups stacked in x_stacked, in order: a frame holding only some of the fitted groups stacks only those.
        groups_stacked = _groups_in_data(data, self.group_col_, self.unique_groups_, self.norm_params_)
        return x_stacked, y_stacked, groups_stacked

    def _warn_extrapolation(self, data: pd.DataFrame, x_stacked: torch.Tensor) -> None:
        r"""
        Warns, once per model and feature, when a forecast leaves the trained range of a non-linear effect, or holds a categorical
        level that training never saw (``TAMExtrapolationWarning``). Features are normalised per group, so the range is the group's own.
        """
        warned = self.__dict__.setdefault("_extrapolation_warned_", set())
        features = self.features_config_['features'] if self.features_config_ else []
        nonlinear = extrapolating_features(self.effects_list_)
        for j, name in enumerate(features):
            if name not in nonlinear or name in warned:
                continue
            beyond = (x_stacked[..., j].abs() - 1.0).clamp(min=0)
            outside = beyond > EXTRAPOLATION_TOLERANCE
            if outside.any():
                warned.add(name)
                warn_extrapolation(name,
                    f"'{name}' leaves the range seen in training on {outside.double().mean().item():.1%} of the rows, by up to "
                    f"{beyond.max().item():.2f} half-ranges: its non-linear effect is extrapolated and can be far off. "
                    f"Set extrapolate='constant' on the term to hold it at the edge of the trained range.", stacklevel=4)
        seen = getattr(self, "_seen_levels_", None) or {}
        for name, levels in seen.items():
            key = f"{name} (levels)"
            if key in warned or name not in data.columns:
                continue
            unseen = sorted(set(np.unique(data[name].dropna().to_numpy()).tolist()) - levels)
            if unseen:
                warned.add(key)
                warn_extrapolation(key,
                    f"'{name}': level(s) {unseen[:10]} not seen in training; the categorical effect has no estimate for them "
                    f"(a code that is none of the levels has no categorical effect: the prediction falls back on the other terms).", stacklevel=4)

    def _build_design_matrix(self, x_data: torch.Tensor) -> torch.Tensor:
        """Builds the global design matrix."""
        feature_names = self.features_config_['features'] if self.features_config_ else None
        return build_phi_from_effects(x_data, self.effects_list_, feature_columns=feature_names)

    def _build_penalty_matrix(self) -> torch.Tensor:
        """Builds the global penalty matrix."""
        return build_penalty_from_effects(self.effects_list_)

    def _build_loss_matrix(self) -> torch.Tensor:
        """Builds the Identity loss matrix (Standard Least Squares)."""
        return torch.eye(1, device=TORCH_DEVICE, dtype=torch.get_default_dtype())
    
#: <decompose_pred>
    def decompose_prediction(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Decomposes the prediction into additive components per feature.

        Args:
            data: DataFrame containing input features.

        Returns:
            DataFrame with original data and additional 'effect_feature' columns.
        """
        data = _ensure_dummies(data, self.group_col_, self.date_col_)

        if self.coefficients_ is None:
            raise RuntimeError("Model must be fitted first.")

        required_cols = self.features_config_['features'] + [self.group_col_, self.date_col_]
        _check_features(dataset=data, required_features=required_cols)
        _check_known_groups(data, self.group_col_, self.unique_groups_)
        
        mask, balanced_data = _balance_groups(
            dataset=data, group_col=self.group_col_, date_col=self.date_col_, method="fill"
        )

        x_predict, _, groups_stacked = self._prepare_data(balanced_data)
        
        final_decomposed_effects = smart_decompose(
            x_predict, _coefficients_of_groups(self.coefficients_, groups_stacked, self.unique_groups_), self.effects_list_
        )

        decomposed_df = _reassemble_decomposed_predictions(
            balanced_data, final_decomposed_effects, self.group_col_, groups_stacked, date_col=self.date_col_
        )

        decomposed_df = _cleanup_dummies(decomposed_df[mask], self.group_col_, self.date_col_)
        for name in getattr(self, "_offset_features_", []):
            decomposed_df[f"effect_offset_{name}"] = decomposed_df[name].to_numpy(dtype=float)
        return decomposed_df
#: </decompose_pred>
    
    # --- Grid Search (Coordinate Descent) Methods ---
    
    def _parse_grid_axes(self, grid_search_config: dict) -> Tuple[Dict[str, list], List[str]]:
        """Identifies which parameters are tokens to be optimized."""
        if not self.is_grid_search_template_:
            return {}, []
        
        tokens = {} 
        for term in self.parsed_terms_:
            for key, val in term['params'].items():
                if isinstance(val, str) and val in grid_search_config:
                    if val not in tokens:
                         tokens[val] = grid_search_config[val]

        token_names = self._extract_recursive_tokens(self.parsed_terms_, grid_search_config)
        search_axes = {tname: grid_search_config[tname] for tname in token_names}

        if token_names:
            print(f"Identified Coordinate Descent axes: {token_names}")
        return search_axes, token_names
    
    def _extract_recursive_tokens(self, terms: List[Dict], grid_config: Dict) -> List[str]:
        """Recursively extracts all token names used in the formula, diving into Tensor Products."""
        tokens = set()
        CONFIG_KEYS = {'basis', 'activation', 'act', 'topo', 'topology', 'op', 'others', 'extrapolate'}
        
        for term in terms:
            for key, val in term['params'].items():
                if isinstance(val, str) and val in grid_config:
                    tokens.add(val)
                if isinstance(key, str) and key in grid_config:
                    tokens.add(key)
            
            if term['type'] == 'te':
                for sub_term_raw in term['params'].keys():
                    if not sub_term_raw: continue
                    token_name_regex = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)')
                    dummy_token_map = {t: t for t in grid_config.keys()}
                    
                    def find_token_name(match):
                        token = match.group(1)
                        if token in dummy_token_map:
                            return token
                        return ""
                        
                    found_tokens = [t for t in token_name_regex.findall(sub_term_raw) if t in grid_config]
                    tokens.update(found_tokens)
                    
        return sorted(list(tokens))

    def _build_combo_from_tokens(self, token_values: Dict[str, Any], data_info: Optional[Dict[str, Any]] = None) -> Dict:
        """Creates a concrete configuration from tokens."""
        effects_list = create_effects_from_parsed_terms(
            self.parsed_terms_,
            token_values,
            self.default_alpha_p_,
            data_info=data_info
        )
        return {
            "effects_list": effects_list,
            "token_values": token_values
        }

    def _evaluate_combination(
        self, combo: dict, 
        x_train: torch.Tensor, y_train: torch.Tensor,
        x_val: torch.Tensor, y_val: torch.Tensor,
        num_samples_train: int, loss_L_star_L: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Evaluates a single hyperparameter configuration on validation data.
        
        Uses dynamic memory chunking to prevent OutOfMemory errors when 
        building the design matrix for massive models.
        """
        try:
            effects_list = combo["effects_list"]
            penalty_M_star_M = build_penalty_from_effects(effects_list)
            self.effects_list_ = effects_list 
            
            strategy = getattr(self, "_reweighting_strategy_", None)
            if strategy is not None:
                # A GLM loss: reweighted fit of the candidate, scored on the response scale (inverse link applied).
                self._reweighting_strategy_ = build_strategy(self.loss_, **self._loss_kwargs_)
                self._solve_prepared(x_train, y_train)
                adaptive_coeffs_batch = self.coefficients_
                mean = self._reweighting_strategy_.inverse_link(_linear_predictor(self, x_val, adaptive_coeffs_batch).cpu())
                rmse = torch.sqrt(torch.mean((mean - y_val.cpu().to(mean.dtype)) ** 2))
            else:
                offset_train, offset_val = self._offset_of(x_train), self._offset_of(x_val)
                adaptive_coeffs_batch = smart_solve(
                    x_data=x_train,
                    y_data=y_train if offset_train is None else y_train - offset_train.to(y_train.dtype),
                    effects_list=effects_list,
                    penalty_matrix=penalty_M_star_M,
                    loss_matrix=loss_L_star_L,
                    num_samples=num_samples_train
                )

                rmse = smart_evaluate_rmse(x_val, y_val if offset_val is None else y_val - offset_val.to(y_val.dtype),
                                           adaptive_coeffs_batch, effects_list)

            return rmse, adaptive_coeffs_batch

        except Exception:
            return torch.tensor(float('inf')), None

#: <grid_search_logic>
    def grid_search_fit(
            self, 
            data_train: pd.DataFrame, 
            data_val: pd.DataFrame, 
            grid_search_config: dict
        ):
            """
            Finds optimal hyperparameters via Multi-Start Coordinate Descent.

            Args:
                data_train (pd.DataFrame): Training data.
                data_val (pd.DataFrame): Validation data for scoring.
                grid_search_config (dict): Dictionary mapping tokens to lists of values.

            Returns:
                StaticTAM: A new fitted model with optimal parameters.
            """
            print("--- Starting Grid Search (Multi-Start Coordinate Descent) ---")
            
            data_train = self._transformed_frame(data_train, estimate=True)
            data_val = self._transformed_frame(data_val)
            transformed_train = data_train
            data_train = _ensure_dummies(data_train, self.group_col_, self.date_col_)
            data_val = _ensure_dummies(data_val, self.group_col_, self.date_col_)
            
            temp_model = StaticTAM(self.formula_, self.group_col_, self.date_col_, loss=self.loss_, loss_kwargs=self._loss_kwargs_)
            temp_model.features_config_ = self.features_config_
            temp_model.target_col_ = self.target_col_
            
            required_cols_tr = self.features_config_['features'] + [self.group_col_, self.target_col_, self.date_col_]
            _check_features(dataset=data_train, required_features=required_cols_tr)
            _, balanced_data_train = _balance_groups(dataset=data_train, group_col=self.group_col_, date_col=self.date_col_, method="drop")
            
            required_cols_val = self.features_config_['features'] + [self.group_col_, self.target_col_, self.date_col_]
            _check_features(dataset=data_val, required_features=required_cols_val)
            _, balanced_data_val = _balance_groups(dataset=data_val, group_col=self.group_col_, date_col=self.date_col_, method="drop")
            
            x_train, y_train, unique_groups = temp_model._prepare_data(balanced_data_train, self.target_col_, ignore_template_check=True)
            x_val, y_val, _ = temp_model._prepare_data(balanced_data_val, self.target_col_, ignore_template_check=True)
            
            num_samples_train = x_train.shape[1]
            loss_L_star_L = torch.eye(1, device=TORCH_DEVICE, dtype=torch.get_default_dtype()) 

            search_axes, token_names = self._parse_grid_axes(grid_search_config)
            data_info = self._get_data_info(balanced_data_train)

            if not token_names:
                print("No grid tokens found. Fitting single configuration.")
                combo = self._build_combo_from_tokens({}, data_info=data_info)
                rmse, coeffs = self._evaluate_combination(
                    combo, x_train, y_train, x_val, y_val, num_samples_train, loss_L_star_L
                )
                optimal_params_combo, optimal_coeffs, min_global_rmse = combo, coeffs, rmse
            else:
                start_points = [
                    {"name": "Conservative", "tokens": {t: max(vals) if ('ap' in t or 'lambda_p' in t) else min(vals) for t, vals in search_axes.items()}},
                    {"name": "Median", "tokens": {t: vals[len(vals)//2] for t, vals in search_axes.items()}},
                    {"name": "Aggressive", "tokens": {t: min(vals) if ('ap' in t or 'lambda_p' in t) else max(vals) for t, vals in search_axes.items()}}
                ]

                global_best_rmse = float('inf')
                global_best_combo = None
                global_best_coeffs = None

                for strategy in start_points:
                    print(f"\n=== Strategy: {strategy['name']} Start ===")
                    current_best_tokens = strategy["tokens"].copy()
                    
                    complete_token_map = {}
                    for tname, tvals in search_axes.items():
                        if tname in current_best_tokens:
                            complete_token_map[tname] = current_best_tokens[tname]
                        else:
                            complete_token_map[tname] = tvals[0]
                    
                    current_best_tokens = complete_token_map.copy()

                    start_combo = self._build_combo_from_tokens(current_best_tokens, data_info=data_info)
                    current_rmse, current_coeffs = self._evaluate_combination(
                        start_combo, x_train, y_train, x_val, y_val, num_samples_train, loss_L_star_L
                    )
                    current_optimal_combo = start_combo

                    if current_rmse >= float('inf'):
                        continue

                    cycle = 0
                    while True:
                        cycle += 1
                        has_improved_in_cycle = False
                        
                        for token_name in token_names:
                            best_val_for_axis = current_best_tokens[token_name]
                            original_val = best_val_for_axis
                            possible_values = search_axes[token_name]
                            
                            for value in possible_values:
                                if value == original_val: continue 
                                tokens_to_test = current_best_tokens.copy()
                                tokens_to_test[token_name] = value
                                
                                combo = self._build_combo_from_tokens(tokens_to_test, data_info=data_info)
                                rmse, coeffs = self._evaluate_combination(
                                    combo, x_train, y_train, x_val, y_val, num_samples_train, loss_L_star_L
                                )

                                if rmse < current_rmse:
                                    current_rmse = rmse
                                    current_optimal_combo = combo
                                    current_coeffs = coeffs
                                    best_val_for_axis = value
                                    has_improved_in_cycle = True
                            
                            current_best_tokens[token_name] = best_val_for_axis
                        
                        print(f"  Cycle {cycle} | Current RMSE: {current_rmse:.4f}")
                        if not has_improved_in_cycle or cycle >= 5: break
                    
                    if current_rmse < global_best_rmse:
                        print(f"  >>> New Global Best found by {strategy['name']}! ({current_rmse:.4f})")
                        global_best_rmse = current_rmse
                        global_best_combo = current_optimal_combo
                        global_best_coeffs = current_coeffs

                optimal_params_combo = global_best_combo
                optimal_coeffs = global_best_coeffs
                min_global_rmse = global_best_rmse

            print("-" * 30)
            print(f"Grid search complete. Optimal Validation RMSE found: {min_global_rmse:.2f}")
            
            if optimal_params_combo is None:
                raise RuntimeError("Grid search failed to find any valid configuration.")

            print(f"Best tokens: {optimal_params_combo.get('token_values', 'N/A')}")

            model = self.__class__(
                formula=self.formula_, 
                group_col=self.group_col_,
                date_col=self.date_col_,
                loss=self.loss_,
                loss_kwargs=self._loss_kwargs_,
                target_transform=self._plain_transform_ if self._transform_active() else None,
                transform_kwargs=self._plain_kwargs_,
                _internal_effects_list=optimal_params_combo['effects_list'],
                _internal_features_config=self.features_config_ 
            )
            
            model.coefficients_ = optimal_coeffs
            model.validation_rmse_ = float(min_global_rmse)
            model.dispersion_ = getattr(self._reweighting_strategy_, "dispersion", None)
            model.target_col_ = self.target_col_ 
            model.norm_params_ = temp_model.norm_params_
            model._vocabulary_ = getattr(temp_model, "_vocabulary_", {})
            model.unique_groups_ = unique_groups
            model._offset_features_ = list(getattr(self, "_offset_features_", []))
            if self._transform_active():
                model._plain_params_ = self._plain_params_
                model._finish_transform_fit(transformed_train)

            return model
#: </grid_search_logic>
    
    def _auto_fit_reweighted(self, x_train, y_train, loss_L, alpha_p_bounds, number_of_steps, alpha_p_list, gamma, verbose):
        """Working-weight GCV of a GLM loss: fit, score the working system, repeat until the smoothing parameters settle.

        One pass fits the reweighted model at the current smoothing parameters, takes its working response z and Fisher
        weights w, and lets the GCV search choose the parameters of the weighted system (sqrt(w) Phi, sqrt(w) z). The final
        fit is the reweighted fit at the selected parameters, so ``coefficients_`` and ``dispersion_`` are those of ``fit()``.
        """
        strategy = self._reweighting_strategy_
        previous = np.array([c for e in self.effects_list_ for c in e.penalty_coordinates()], dtype=np.float64)
        best_lambda_ps, gcv_score = previous, float("nan")
        for _ in range(_GCV_REWEIGHTING_PASSES):
            self._solve_prepared(x_train, y_train)
            eta = _linear_predictor(self, x_train, self.coefficients_)
            working_response, weights = strategy.working_response_and_weights(y_train.to(torch.get_default_dtype()), eta)
            weights = weights / weights.mean().clamp_min(_TINY)
            offset = self._offset_of(x_train)
            if offset is not None:
                working_response = working_response - offset.to(working_response.dtype)
            _, best_lambda_ps, gcv_score = smart_solve_gcv(
                x_data=x_train,
                y_data=working_response,
                effects_list=self.effects_list_,
                loss_matrix=loss_L,
                alpha_p_bounds=alpha_p_bounds,
                number_of_steps=number_of_steps,
                alpha_p_list=alpha_p_list,
                gamma=gamma,
                verbose=verbose,
                sample_weights=weights,
            )
            moved = float(np.max(np.abs(np.log10(best_lambda_ps) - np.log10(previous))))
            previous = np.asarray(best_lambda_ps, dtype=np.float64)
            if moved < _GCV_REWEIGHTING_TOL:
                break
        self._solve_prepared(x_train, y_train)
        return best_lambda_ps, gcv_score

#: <auto_fit>
    def auto_fit(
        self, 
        data_train: pd.DataFrame, 
        alpha_p_bounds: Tuple[float, float] = (-30.0, 6.0),
        number_of_steps: int = 10,
        alpha_p_list: Optional[List[float]] = None,
        gamma: float = 1.4,
        verbose: bool = False
    ) -> 'StaticTAM':
        """
        Automatically trains the model by finding the optimal regularization per effect.
        
        Uses Generalized Cross Validation (GCV) with Multiple Smoothing Parameter (MSP)
        estimation to balance model fit (bias) and complexity (variance) without 
        requiring a validation set. 
        
        It utilizes a Coordinate Descent algorithm starting from the formula's initial values.
        
        Args:
            data_train: Training DataFrame.
            alpha_p_bounds: Search range constraints for local steps (e.g., -30.0 to 6.0).
            number_of_steps: Number of subdivisions within alpha_p_bounds to define the step size.
            alpha_p_list: Explicit list of log10(lambda_p) coordinates to test. If provided,
                        alpha_p_bounds and number_of_steps are ignored.
            gamma: Inflation factor for the effective degrees of freedom. Values > 1.0 
                   (typically 1.4 to 1.5) force smoother models, preventing GCV 
                   from overfitting when data has autocorrelated errors.
            verbose: print verbose details if True
            
        Returns:
            self: The trained model instance with optimal lambda_ps applied.
        """
        print("--- Auto-Fitting with MSP-GCV (Multiple Smoothing Parameters) ---")

        data_train = self._transformed_frame(data_train, estimate=True)
        transformed_train = data_train
        data_train = _ensure_dummies(data_train, self.group_col_, self.date_col_)
        
        _, balanced_data = _balance_groups(
            dataset=data_train, group_col=self.group_col_, date_col=self.date_col_, method="drop"
        )
        x_train, y_train, self.unique_groups_ = self._prepare_data(
            balanced_data, target_col=self.target_col_
        )
        
        loss_L = self._build_loss_matrix()
        
        if getattr(self, "_reweighting_strategy_", None) is not None:
            best_lambda_ps, gcv_score = self._auto_fit_reweighted(
                x_train, y_train, loss_L, alpha_p_bounds, number_of_steps, alpha_p_list, gamma, verbose
            )
        else:
            offset = self._offset_of(x_train)
            self.coefficients_, best_lambda_ps, gcv_score = smart_solve_gcv(
                x_data=x_train,
                y_data=y_train if offset is None else y_train - offset.to(y_train.dtype),
                effects_list=self.effects_list_,
                loss_matrix=loss_L,
                alpha_p_bounds=alpha_p_bounds,
                number_of_steps=number_of_steps,
                alpha_p_list=alpha_p_list,
                gamma=gamma,
                verbose=verbose
            )
        
        print(f"\nFinal GCV Score: {gcv_score:.4f}")
        print("Optimal lambda_ps found per effect (per margin for a tensor product):")
        position = 0
        for effect in self.effects_list_:
            count = effect.n_penalty_coordinates
            values = best_lambda_ps[position:position + count]
            position += count
            effect.set_penalty_coordinates([float(v) for v in values])
            if isinstance(effect, TensorProductEffect):
                start = 0
                for margin in effect.effects:
                    for part, value in enumerate(values[start:start + margin.n_penalty_coordinates]):
                        label = margin.feature_name if margin.n_penalty_coordinates == 1 else f"{margin.feature_name}#{part + 1}"
                        print(f" - te({', '.join(m.feature_name for m in effect.effects)})[{label}]: {value:.2e} (log10 = {np.log10(value):.2f})")
                    start += margin.n_penalty_coordinates
            else:
                for part, value in enumerate(values):
                    label = effect.feature_name if len(values) == 1 else f"{effect.feature_name}#{part + 1}"
                    print(f" - {label}: {value:.2e} (log10 = {np.log10(value):.2f})")

        if self._transform_active():
            self._finish_transform_fit(transformed_train)
        return self
#: </auto_fit>
    
    @staticmethod
    def show_syntax():
        """
        Displays a quick reference guide for formula syntax in the console.
        """

        data = [
            {
                "Token": "l(x)", 
                "Effect": "Linear / Ridge", 
                "Syntax Example": "l(trend, scaled=3.14)",
                "Specific Params": "scaled (data scaling)"
            },
            {
                "Token": "s(x)", 
                "Effect": "P-Spline", 
                "Syntax Example": "s(temp, k=12, deg=3, p=2)",
                "Specific Params": "k (knots), deg (degree), p (penalty order), m_pen (locally adaptive penalty, m smoothing parameters)"
            },
            {
                "Token": "f(x)", 
                "Effect": "Fourier", 
                "Syntax Example": "f(hour, m=6, s=1, cyclic=True, period='auto')",
                "Specific Params": "m (harmonics), s (smoothness), cyclic (bool), period (None: training range; 'auto': first to last value + one step; (low, high): given)"
            },
            {
                "Token": "offset(x)",
                "Effect": "Exposure offset (fixed coefficient 1)",
                "Syntax Example": "offset(log_exposure)",
                "Specific Params": "none (raw column, no parameter)",
                "Usage": "Rates: counts over a variable effort, with loss='poisson' or 'negative_binomial'"
            },
            {
                "Token": "c(x)", 
                "Effect": "Categorical", 
                "Syntax Example": "c(day, topo='ordinal', p_order=1)",
                "Specific Params": "n_cat (auto-inferred), topo ('nominal'/'ordinal'), p_order"
            },
            {
                "Token": "p(x)", 
                "Effect": "Chebyshev", 
                "Syntax Example": "p(time, deg=5, s=0)",
                "Specific Params": "deg (degree), s (smoothness)",
                "Usage": "Secular trends, stable extrapolation (Anti-Runge)"
            },
            {
                "Token": "rbf(x)", 
                "Effect": "RBF / Matérn Kernel", 
                "Syntax Example": "rbf(Lat, others='Lon', n_centers=50)",
                "Specific Params": "n_centers, gamma, nu, others (multivariate)"
            },
            {
                "Token": "n(x)", 
                "Effect": "Neural Network", 
                "Syntax Example": "n(Income, others='Age', n_neurons=500, seed=42)",
                "Specific Params": "n_neurons, act ('relu'/'cos'/'tanh'), n_hidden_layers, seed, others"
            },
            {
                "Token": "te(a, b)", 
                "Effect": "Tensor Product", 
                "Syntax Example": "te(s(Temp), f(Hour))",
                "Specific Params": "List of functional sub-effects"
            },
            {
                "Token": "w(x)", 
                "Effect": "Wavelet (Ricker)", 
                "Syntax Example": "w(signal, n_scales=5, n_locations=20)",
                "Specific Params": "n_scales, n_locations"
            },
            {
                "Token": "phys(x)", 
                "Effect": "Physics (PIKL/ODE)", 
                "Syntax Example": "phys(t, basis='spline', k=20, D2=1.0)",
                "Specific Params": "basis, k/n_coeffs, D{n} (derivative weights)"
            },
            {
                "Token": "pid(x)", 
                "Effect": "PID (Control)", 
                "Syntax Example": "pid(y_lag, w=7, d_pen=10.0)",
                "Specific Params": "w (rolling window), d_pen (derivative stiffness)"
            },
            {
                "Token": "t(x)", 
                "Effect": "Tree / Random Forest", 
                "Syntax Example": "t(x, n_trees=50, sp_alpha=0.5, split_strategy='quantile')",
                "Specific Params": "n_trees, max_depth, max_leaves, sp_alpha, split_strategy, seed, others"
            },
            {
                "Token": "lt(x)", 
                "Effect": "Linear Tree (VC Model)", 
                "Syntax Example": "lt(x_part, slope='x_slope', max_leaves=8, sp_alpha=0.5)",
                "Specific Params": "slope, max_depth, max_leaves, sp_alpha, split_strategy, seed, others"
            }
        ]
        
        # Using .fillna("") ensures the 'Usage' column doesn't print 'NaN' for effects without it.
        df = pd.DataFrame(data).fillna("")
        
        print("\n" + "=" * 120)
        print(" " * 45 + "TAM FORMULA SYNTAX GUIDE")
        print("=" * 120)
        print("GLOBAL ARGUMENTS (Applicable to ALL effects):")
        print("  * 'ap'          : log10(lambda_p) regularization strength (e.g., ap=-5 means lambda_p = 1e-5).")
        print("  * 'extrapolate' : OOD behavior ('continue', 'constant', 'linear', 'saturation').")
        print("                  Note: Defaults vary logically by effect topology (e.g., Splines default to 'linear',")
        print("                        Chebyshev defaults to 'saturation', Trees default to 'continue').\n")
        
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1200):
            print(df.to_string(index=False, justify='left'))
        print("=" * 120 + "\n")