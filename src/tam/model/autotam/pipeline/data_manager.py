# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-FileContributor: Amaury Durand
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Data Manager for Automated TAM (AutoTAM).

Handles data topology, semantic parsing, chronological splitting, 
and feature engineering for the AutoTAM pipeline.
"""

#: <data_manager_imports>
import logging
import re
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Tuple, Union
from .context import PipelineContext

from tam.common.utils import split_args_respecting_parentheses
from tam.model.autotam.parser import FormulaParser, canonicalize_term
from tam.model.autotam.population_nodes import MAX_ACTIVE_EFFECTS_PER_FEATURE, MAX_TENSOR_TERMS
from tam.model.autotam.data_profiler import DataProfiler
from tam.model.autotam.feature_engineer import FeatureEngineer
from tam.model.autotam.effect_selector import EffectSelector
from tam.model.autotam.failures import log_failure
from tam.model.autotam.anchor import add_anchor_lag, anchor_column, natural_season, resolve_season
from tam.model.autotam.calendar_features import MIN_STEPS_PER_DAY_FOR_HOLIDAYS, add_calendar, calendar_names, guarded_calendar_names
from tam.model.autotam.spectral_profile import steps_per_day
from tam.model.autotam.regimes import add_regime_column, fit_regimes
from tam.model.autotam.crisis_calendar import active_events, add_crisis, resolve_events
from tam.model.autotam.floor import FLOOR_COLUMN, season_position
from tam.model.autotam.target_profile import mask_sentinels, profile_target
from tam.model.autotam.sentinel_cleaning import SentinelCleaner
from tam.model.autotam.structure_columns import random_effect_columns, spatial_pairs
logger = logging.getLogger(__name__)
#: </data_manager_imports>

#: <data_manager_past_covariates>
def normalize_past_covariates(past_covariates: Optional[Dict[str, List[int]]], columns, target: str, date_col: Optional[str],
                              group_col: Optional[str], features, target_lags: List[int]) -> Dict[str, List[int]]:
    """Validates the declaration of the past-only covariates and returns it with sorted, distinct lags.

    A past-only covariate is known up to the forecast origin only: its value at a forecast row is not known, so only its lags of
    at least the horizon are legitimate features. The horizon is the smallest lag of the target when the target has lags.

    Args:
        past_covariates: Column name -> lag orders, as given to ``AutoTAM.fit``.
        columns: Columns of the training data.
        target: Name of the target column.
        date_col: Name of the date column.
        group_col: Name of the group column.
        features: Features named in the formula pipeline macro.
        target_lags: Lag orders of the target (explicit or parsed from the formula).

    Returns:
        The validated declaration, ``{}`` when ``past_covariates`` is ``None``.

    Raises:
        ValueError: A column is unknown, is the target, the date or the group column, or is also a feature of the formula; a lag is not an integer,
            is below 1, or is below the horizon.
    """
    if not past_covariates:
        return {}
    horizon = min(target_lags) if target_lags else 1
    clean: Dict[str, List[int]] = {}
    for column, lags in past_covariates.items():
        if column not in columns:
            raise ValueError(f"Past covariate '{column}' is not a column of the training data.")
        if column in (target, date_col, group_col):
            raise ValueError(f"Past covariate '{column}' is the target, the date or the group column: it cannot be a covariate.")
        if column in set(features):
            raise ValueError(
                f"'{column}' is declared as a past covariate: it is known only up to the forecast origin, so it cannot be a feature itself. "
                "Remove it from the pipeline macro; its lags are offered to the search."
            )
        if isinstance(lags, (int, np.integer)):
            lags = [lags]
        lags = list(lags or [])
        if not lags:
            raise ValueError(f"Past covariate '{column}' needs at least one lag order.")
        for lag in lags:
            if isinstance(lag, bool) or not isinstance(lag, (int, np.integer)):
                raise ValueError(f"The lag orders of past covariate '{column}' must be integers, got {lag!r}.")
            if lag < 1:
                raise ValueError(f"The lag orders of past covariate '{column}' must be at least 1, got {lag}.")
            if lag < horizon:
                raise ValueError(
                    f"Lag {lag} of past covariate '{column}' is shorter than the horizon ({horizon}, the smallest lag of the target): "
                    "its value would not be known at the forecast origin."
                )
        clean[column] = sorted({int(lag) for lag in lags})
    return clean


def add_past_covariate_lags(df: pd.DataFrame, past_covariates: Dict[str, List[int]], group_col: Optional[str]) -> None:
    """Adds ``<column>_lag_<order>`` for each past covariate, inside each group, the way the lags of the target are built (in place)."""
    for column, lags in past_covariates.items():
        for lag in lags:
            if group_col and group_col in df.columns:
                df[f"{column}_lag_{lag}"] = df.groupby(group_col)[column].shift(lag)
            else:
                df[f"{column}_lag_{lag}"] = df[column].shift(lag)
#: </data_manager_past_covariates>

#: <data_manager_missing_target>
def keep_forecast_targets_missing(df_aug: pd.DataFrame, df_observed: pd.DataFrame, ctx: PipelineContext) -> pd.DataFrame:
    """Puts back the missing target on the rows after the last observed one.

    The continuity step of the data profiler forward-fills at most three missing values of every column, the target included: the first three
    forecast rows of a frame whose target is not known would carry an invented target, and the Kalman and adaptive experts would take it for an
    observation. The rows dated after the last observed target (over all groups) get their missing target back.

    Args:
        df_aug: The augmented frame, after the continuity step.
        df_observed: The frame before it, with the target as it was given.
        ctx: The pipeline context (target and date column).

    Returns:
        ``df_aug``, with a missing target on the forecast rows.
    """
    target, date_col = ctx.target, ctx.date_col
    if not date_col or target not in df_aug.columns or target not in df_observed.columns or date_col not in df_aug.columns or date_col not in df_observed.columns:
        return df_aug
    observed = df_observed.loc[df_observed[target].notna(), date_col]
    if observed.empty:
        return df_aug
    after = pd.to_datetime(df_aug[date_col]) > pd.to_datetime(observed).max()
    if not after.any():
        return df_aug
    out = df_aug.copy()
    out.loc[after, target] = np.nan
    return out
#: </data_manager_missing_target>

#: <data_manager_rolling>
def rolling_origin_folds(pool: pd.DataFrame, date_col: str, k: int, rows: int) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Rolling-origin folds: the last ``k * rows`` dates of ``pool`` in ``k`` consecutive blocks of ``rows`` dates, expanding training sets.

    A block is validated on a model trained on every row dated before it, so a fold has the same look-ahead as a forecast of ``rows`` steps
    from its origin, for every group at once. When the pool is too short to leave a third of its dates (at least two blocks) for the first
    training set, fewer folds are made; ``[]`` when not even one fits.

    Args:
        pool: Training rows (fit and dev), with ``date_col``.
        date_col: Name of the date column.
        k: Number of folds asked for.
        rows: Dates per fold (the horizon).

    Returns:
        ``[(train, validation), ...]`` in time order.
    """
    stamps = pd.to_datetime(pool[date_col])
    dates = np.sort(stamps.unique())
    minimum_train = max(2 * rows, len(dates) // 3)
    folds_possible = min(k, (len(dates) - minimum_train) // rows)
    if folds_possible < 1:
        return []
    folds = []
    for i in range(folds_possible):
        start = len(dates) - (folds_possible - i) * rows
        block = dates[start:start + rows]
        folds.append((pool[stamps < dates[start]].copy(), pool[stamps.isin(block)].copy()))
    return folds
#: </data_manager_rolling>

#: <data_manager_class>
class DataManager:
    """
    Central data orchestrator.
    Manages the lifecycle of datasets through parsing, cleaning, and augmentation.
    """
    
#: <data_manager_init>
    def __init__(
        self, 
        formula: str, 
        lags: Optional[List[int]] = None,
        mandatory_variables: Optional[List[str]] = None,
        train_fraction: float = 0.70,
        dev_fraction: float = 0.15,
        horizon: Optional[int] = None,
        calendar_features: bool = False,
        country: Optional[str] = None,
        subdivision: Optional[str] = None,
        regimes: bool = False,
        crisis_calendar: Any = None,
        extended_search: bool = False,
        rolling_folds: Optional[int] = None,
        pooled_floor: bool = False,
        max_cv_folds: Optional[int] = None,
        clean_sentinels: bool = False,
        calendar_guards: bool = False,
        offset_col: Optional[str] = None,
        strong_regularization: bool = False,
        extended_effects: bool = False,
        small_sample: bool = False,
        spatial_pairs: bool = False,
        random_effects: bool = False,
        smooth_monotonic: bool = False,
        cross_sectional: bool = False,
        physical_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        """
        Initializes the DataManager, parsing the user formula and populating mandatory terms.

        Args:
            formula (str): High-level AutoTAM formula with embedded mandatory terms and a
                pipeline macro (e.g., 'load ~ s(temp, k=10) + AutoPipe(temp, humidity)').
            lags (Optional[List[int]]): Explicit lag orders to inject into the dataset.
            mandatory_variables (Optional[List[str]]): Variables required in all models.
            train_fraction (float): Proportion of training data allocated to model fitting.
            dev_fraction (float): Proportion of training data allocated to hyperparameter tuning.
            horizon (Optional[int]): Steps ahead to forecast; ``None`` takes the smallest lag. Lags shorter than it are refused at ``prepare``.
        """
        if horizon is not None and (int(horizon) != horizon or horizon < 1):
            raise ValueError(f"horizon must be a positive whole number of steps, got {horizon!r}.")
        self.horizon = None if horizon is None else int(horizon)
        self.calendar_features = bool(calendar_features)
        self.regimes = bool(regimes)
        self.crisis_events = resolve_events(crisis_calendar)
        self.pooled_floor = bool(pooled_floor)
        if rolling_folds is not None and (isinstance(rolling_folds, bool) or int(rolling_folds) != rolling_folds or rolling_folds < 1):
            raise ValueError(f"rolling_folds must be None or a positive whole number, got {rolling_folds!r}.")
        self.rolling_folds = None if rolling_folds is None else int(rolling_folds)
        if max_cv_folds is not None and (isinstance(max_cv_folds, bool) or int(max_cv_folds) != max_cv_folds or max_cv_folds < 1):
            raise ValueError(f"max_cv_folds must be None or a positive whole number, got {max_cv_folds!r}.")
        self.max_cv_folds = None if max_cv_folds is None else int(max_cv_folds)
        self.clean_sentinels = bool(clean_sentinels)
        self.calendar_guards = bool(calendar_guards)
        self.cross_sectional = bool(cross_sectional)
        self.small_sample = bool(small_sample)   # fewer folds when the sample is tiny (see generate_cv_folds)
        self.offset_col = offset_col
        self.find_spatial_pairs = bool(spatial_pairs)
        self.find_random_effects = bool(random_effects)
        self.physical_bounds = dict(physical_bounds or {})
        self.country = country
        self.subdivision = subdivision
        self.formula = formula
        self.user_lags = lags or []
        if isinstance(mandatory_variables, str):
            mandatory_variables = [mandatory_variables]
        self.mandatory_variables = mandatory_variables or []
        self.train_fraction = train_fraction
        self.dev_fraction = dev_fraction
        
        self.parser = FormulaParser()
        self.formula_config = self.parser.parse(self.formula)
        self.mandatory_terms = self.formula_config["mandatory_terms"]
        self.profiler = DataProfiler()
        self.engineer = FeatureEngineer(collinearity_threshold=0.95)  # spec I: bar |rho| > 0.95
        self.selector = EffectSelector(extended=extended_search, smooth_monotonic=smooth_monotonic, integer_discrete=cross_sectional, strong_regularization=strong_regularization,
                                      extended_effects=extended_effects)
#: </data_manager_init>

#: <data_manager_prepare>
    def prepare(
        self, 
        df_train: Optional[pd.DataFrame] = None, 
        df_fit: Optional[pd.DataFrame] = None, 
        df_dev: Optional[pd.DataFrame] = None, 
        df_val: Optional[pd.DataFrame] = None, 
        date_col: Optional[str] = None, 
        group_col: Optional[str] = None,
        validation_strategy: str = 'auto',
        past_covariates: Optional[Dict[str, List[int]]] = None,
        seasonal_anchor: Union[bool, str] = False,
        season: Optional[int] = None,
        replace_sentinels: bool = False
    ) -> PipelineContext:
        """
        Executes the data preparation phase and populates the pipeline context.
        """
        if df_train is None and (df_fit is None or df_dev is None or df_val is None):
            raise ValueError("Provide df_train OR explicit df_fit, df_dev, df_val folds.")

        ctx = PipelineContext(date_col=date_col, group_col=group_col)
        
        ctx.formula_config = self.parser.parse(self.formula, date_col=date_col)
        ctx.target = ctx.formula_config["targets"][0]
        ctx.mandatory_terms = ctx.formula_config["mandatory_terms"]
        self.mandatory_terms = ctx.mandatory_terms
        
        ref_df = df_train if df_train is not None else df_fit
        dataset_features = set(ref_df.columns) - set(ctx.formula_config["targets"])
        if date_col and date_col in dataset_features:
            dataset_features.discard(date_col)
        available_features = set(ctx.formula_config["features"]) | dataset_features

        if self.mandatory_terms:
            seen_canonical = set()
            canonical_to_verbatim = {}
            feature_counts: Dict[str, int] = {}
            tensor_terms_count = 0
            all_mandatory_features = set()

            for term in self.mandatory_terms:
                term_str = term.strip()
                if not term_str:
                    raise ValueError(f"Invalid mandatory term '{term}': Term cannot be empty.")
                if term_str.count("(") != term_str.count(")"):
                    raise ValueError(f"Invalid mandatory term '{term}': unbalanced parentheses.")

                func_match = re.match(r"^\s*([a-zA-Z0-9_]+)\s*\((.*)\)\s*$", term_str)
                if not func_match and term_str != "1":
                    raise ValueError(f"Invalid mandatory term '{term}': malformed term syntax.")

                # Canonicalize term and check for duplicates under parameter-order equivalence
                c_term = canonicalize_term(term)
                if c_term in seen_canonical:
                    raise ValueError(f"Duplicate mandatory term detected in mandatory_terms: '{term}'.")
                seen_canonical.add(c_term)
                canonical_to_verbatim[c_term] = term

                if func_match:
                    eff_type = func_match.group(1).strip()
                    inner_content = func_match.group(2).strip()

                    if eff_type == "te":
                        tensor_terms_count += 1
                        if tensor_terms_count > MAX_TENSOR_TERMS:
                            raise ValueError(
                                f"Number of mandatory tensor terms ({tensor_terms_count}) "
                                f"exceeds MAX_TENSOR_TERMS ({MAX_TENSOR_TERMS})."
                            )
                        sub_parts = split_args_respecting_parentheses(inner_content)
                        for sub_part in sub_parts:
                            sub_part = sub_part.strip()
                            kw_match = re.match(r"^\s*([a-zA-Z0-9_]+)\s*=\s*(.*)$", sub_part)
                            if kw_match:
                                continue

                            sub_match = re.match(r"^\s*([a-zA-Z0-9_]+)\s*\((.*)\)\s*$", sub_part)
                            if sub_match:
                                sub_args = split_args_respecting_parentheses(sub_match.group(2).strip())
                                if not sub_args or not sub_args[0].strip():
                                    raise ValueError(
                                        f"Invalid sub-term in mandatory tensor interaction '{term}': '{sub_part}'."
                                    )
                                sub_feat = sub_args[0].strip()
                            else:
                                sub_feat = sub_part

                            if sub_feat not in available_features:
                                raise ValueError(
                                    f"Feature '{sub_feat}' in mandatory term '{term}' not in available features."
                                )
                            all_mandatory_features.add(sub_feat)
                    else:
                        parts = split_args_respecting_parentheses(inner_content)
                        if not parts or not parts[0].strip():
                            raise ValueError(f"Invalid mandatory term '{term}': missing feature.")
                        feat = parts[0].strip()
                        if feat not in available_features:
                            raise ValueError(f"Feature '{feat}' in mandatory term '{term}' not in available features.")

                        all_mandatory_features.add(feat)
                        feature_counts[feat] = feature_counts.get(feat, 0) + 1
                        if feature_counts[feat] > MAX_ACTIVE_EFFECTS_PER_FEATURE:
                            raise ValueError(
                                f"Feature '{feat}' has {feature_counts[feat]} mandatory terms, "
                                f"which exceeds MAX_ACTIVE_EFFECTS_PER_FEATURE ({MAX_ACTIVE_EFFECTS_PER_FEATURE})."
                            )

            ctx.canonical_to_verbatim_mandatory = canonical_to_verbatim
            ctx.external_mandatory_features = all_mandatory_features - set(ctx.formula_config["features"])
        else:
            ctx.mandatory_terms = []
            ctx.canonical_to_verbatim_mandatory = {}
            ctx.external_mandatory_features = set()

        if self.mandatory_variables:
            ctx.mandatory_variables = self.mandatory_variables
            autopipe_features = set(ctx.formula_config["features"])
            for var in self.mandatory_variables:
                if var not in autopipe_features:
                    raise ValueError(f"Mandatory variable '{var}' not in available features.")

        
        parsed_lags = list(ctx.formula_config.get("lags", {}).values())
        ctx.lags = list(set(self.user_lags + parsed_lags))
        ctx.requested_horizon = self.horizon
        if self.horizon is not None:
            too_short = sorted(lag for lag in ctx.lags if lag < self.horizon)
            if too_short:
                raise ValueError(
                    f"horizon={self.horizon} is not compatible with the lags {too_short} of the target: a lag shorter than the horizon is not "
                    f"known that many steps ahead. Use lags of at least {self.horizon}, or drop the horizon argument."
                )
        horizon_orders = [self.horizon] if self.horizon is not None else ctx.lags  # the smallest of them is the horizon
        ctx.past_covariates = normalize_past_covariates(
            past_covariates, ref_df.columns, ctx.target, date_col, group_col, ctx.formula_config["features"], horizon_orders
        )
        for column, lags in ctx.past_covariates.items():
            for lag in lags:
                if f"{column}_lag_{lag}" not in ctx.formula_config["features"]:
                    ctx.formula_config["features"].append(f"{column}_lag_{lag}")

        if df_train is not None:
            df_proc = df_train.sort_values(date_col) if date_col else df_train
            if self.cross_sectional and not date_col:   # rows with no time order: a split in row order would validate on one end of the covariates
                df_proc = df_proc.sample(frac=1.0, random_state=42)
            n = len(df_proc)
            fit_end = int(n * self.train_fraction)
            dev_end = int(n * (self.train_fraction + self.dev_fraction))

            # Use positional views to prevent a 2x memory duplication overhead when concatenating splits downstream.
            raw_fit = df_proc.iloc[:fit_end]
            raw_dev = df_proc.iloc[fit_end:dev_end]
            raw_val = df_proc.iloc[dev_end:]
            df_all_raw = df_proc.copy()
        else:
            raw_fit, raw_dev, raw_val = df_fit.copy(), df_dev.copy(), df_val.copy()
            has_overlap = (
                not raw_fit.index.is_unique or not raw_dev.index.is_unique or not raw_val.index.is_unique
                or raw_fit.index.intersection(raw_dev.index).size > 0
                or raw_dev.index.intersection(raw_val.index).size > 0
                or raw_fit.index.intersection(raw_val.index).size > 0
            )
            if has_overlap:
                n_fit = len(raw_fit)
                n_dev = len(raw_dev)
                n_val = len(raw_val)
                raw_fit.index = pd.RangeIndex(0, n_fit)
                raw_dev.index = pd.RangeIndex(n_fit, n_fit + n_dev)
                raw_val.index = pd.RangeIndex(n_fit + n_dev, n_fit + n_dev + n_val)
            df_all_raw = pd.concat([raw_fit, raw_dev, raw_val])

        ctx.target_profile = profile_target(pd.concat([raw_fit, raw_dev, raw_val])[ctx.target])
        sentinels = ctx.target_profile["sentinels"]
        if sentinels and replace_sentinels:
            raw_fit, raw_dev, raw_val, df_all_raw = (mask_sentinels(frame, ctx.target, sentinels) for frame in (raw_fit, raw_dev, raw_val, df_all_raw))
            logger.warning("The value(s) %s of the target are missing-value codes: replaced by missing values (replace_sentinels=True).", sentinels)
        elif sentinels:
            logger.warning("The value(s) %s repeat at an extreme of the target and look like missing-value codes (sentinels): they are fitted as values. "
                           "Pass replace_sentinels=True to turn them into missing values.", sentinels)

        if self.clean_sentinels:
            pool = pd.concat([raw_fit, raw_dev, raw_val])
            features = [c for c in ctx.formula_config["features"] if c in pool.columns and c not in (ctx.target, date_col, group_col)]
            cleaner = SentinelCleaner([ctx.target] + features, date_col, group_col, self.physical_bounds, negative_columns=[ctx.target]).fit(pool)
            ctx.sentinel_cleaner = cleaner
            report = cleaner.report(pool)
            raw_fit, raw_dev, raw_val, df_all_raw = (cleaner.apply(frame) for frame in (raw_fit, raw_dev, raw_val, df_all_raw))
            for column, rule in report.items():
                logger.warning("Sentinel cleaning: %d cell(s) of '%s' are missing-data codes or impossible readings (codes %s, range %s, negatives %s): "
                               "replaced by the seasonal median of the training rows.", rule["masked"], column, rule["codes"], rule["bounds"], rule["negatives"])

        empty = [name for name, rows in (("df_fit", raw_fit), ("df_dev", raw_dev), ("df_val", raw_val)) if len(rows) == 0]
        if empty:
            raise ValueError(
                f"AutoTAM needs rows in df_fit, df_dev and df_val, and {', '.join(empty)} is empty "
                f"(rows: fit={len(raw_fit)}, dev={len(raw_dev)}, val={len(raw_val)}). Give a longer history, "
                "or pass df_train= so that AutoTAM sizes the three splits from the data."
            )

        _, ctx.metadata = self.profiler.profile_and_clean(raw_fit, ctx.formula_config, date_col, group_col)

        if seasonal_anchor:
            try:
                ctx.anchor_season = resolve_season(season, ctx.metadata.get("delta_t", "1D"), horizon_orders)
            except ValueError as exc:
                if seasonal_anchor != "auto":   # "auto": the anchor was switched on by a preset, so no usable season only means no anchor
                    raise
                logger.info("Seasonal anchor not used: %s", exc)
            if ctx.anchor_season and anchor_column(ctx.anchor_season) not in ctx.formula_config["features"]:
                ctx.formula_config["features"].append(anchor_column(ctx.anchor_season))

        if date_col:
            df_all_raw = df_all_raw.sort_values(date_col)

        if ctx.lags:
            for d in ctx.lags: 
                if group_col and group_col in df_all_raw.columns:
                    df_all_raw[f"{ctx.target}_lag_{d}"] = df_all_raw.groupby(group_col)[ctx.target].shift(d)
                else: 
                    df_all_raw[f"{ctx.target}_lag_{d}"] = df_all_raw[ctx.target].shift(d)

        add_past_covariate_lags(df_all_raw, ctx.past_covariates, group_col)
        if ctx.anchor_season:
            add_anchor_lag(df_all_raw, ctx.target, ctx.anchor_season, group_col)

        df_all_clean = self.profiler.transform(df_all_raw, ctx.formula_config, date_col) if hasattr(self.profiler, 'transform') else df_all_raw
        self._declare_structure_columns(df_all_clean, ctx, date_col, group_col)
        df_all_aug = self.engineer.engineer_features(df_all_clean, ctx.formula_config, ctx.metadata, date_col)
        
        df_all_aug = self._add_calendar_columns(df_all_aug, ctx)
        df_all_aug = self._add_crisis_columns(df_all_aug, ctx)
        df_all_aug = self._add_regime_column(df_all_aug, ctx)
        df_all_aug = self._add_floor_column(df_all_aug, ctx)
        ctx.df_all_aug = df_all_aug

        if date_col and date_col in df_all_aug.columns and date_col in raw_fit.columns:
            fit_max_date = raw_fit[date_col].max()
            dev_max_date = raw_dev[date_col].max()
            
            ctx.df_fit = df_all_aug[df_all_aug[date_col] <= fit_max_date].copy()
            ctx.df_dev = df_all_aug[(df_all_aug[date_col] > fit_max_date) & (df_all_aug[date_col] <= dev_max_date)].copy()
            ctx.df_val = df_all_aug[df_all_aug[date_col] > dev_max_date].copy()
        else:
            len_fit = len(raw_fit)
            len_dev = len(raw_dev)
            
            ctx.df_fit = df_all_aug.iloc[:len_fit].copy()
            ctx.df_dev = df_all_aug.iloc[len_fit:len_fit + len_dev].copy()
            ctx.df_val = df_all_aug.iloc[len_fit + len_dev:].copy()

        ctx.validation_strategy = validation_strategy
        fold_rows = max(1, int(self.horizon or (min(ctx.lags) if ctx.lags else 1)))
        if self.rolling_folds and fold_rows < 2 and not (group_col and group_col in df_all_aug.columns and df_all_aug[group_col].nunique() > 1):
            # A fold of one row is skipped by the search (a fold needs two rows to be scored): every candidate would be dropped. A one-step
            # horizon is scored on two steps; without a horizon or a lag the folds are a twentieth of the dates.
            fold_rows = max(2, int(pd.concat([ctx.df_fit, ctx.df_dev])[date_col].nunique()) // 20) if (self.horizon is None and not ctx.lags and date_col) else 2
        ctx.cv_folds = self.generate_cv_folds(ctx.df_fit, ctx.df_dev, date_col, validation_strategy, fold_rows=fold_rows)

        look_orders = list(ctx.lags) + [lag for lags in ctx.past_covariates.values() for lag in lags] + ([ctx.anchor_season] if ctx.anchor_season else [])
        max_look = max(look_orders) if look_orders else 24
        if group_col and group_col in df_all_aug.columns:
            ctx.historical_tail = df_all_aug.groupby(group_col).tail(max_look + 1).copy() 
        else:
            ctx.historical_tail = df_all_aug.tail(max_look + 1).copy()

        ctx.formula_config["random_effect_columns"] = list(ctx.random_effect_columns)
        ctx.search_space = self.selector.build_search_space(ctx.df_fit, ctx.formula_config, ctx.metadata)
        if ctx.offset_col:
            ctx.search_space.pop(ctx.offset_col, None)   # the exposure is not a feature: it enters every formula as offset()
        ctx.search_space["mandatory_terms"] = ctx.mandatory_terms
        ctx.search_space["mandatory_variables"] = ctx.mandatory_variables

        reference_df = df_train if df_train is not None else ctx.df_fit
        target_series = reference_df[ctx.target]

        target_var = float(target_series.var())
        target_mean = float(target_series.mean())

        inferred_observation_noise = max(1e-5, target_var * 0.05)
        inferred_P_init = max(1e-2, target_var)
        inferred_offset_boost = abs(target_mean)

        ctx.kalman_priors = {
            "observation_noise_var": inferred_observation_noise,
            "P_init_diag": inferred_P_init,
            "offset_boost": inferred_offset_boost
        }

        return ctx
#: </data_manager_prepare>

#: <data_manager_cv>
    def generate_cv_folds(
        self, 
        df_fit: pd.DataFrame, 
        df_dev: pd.DataFrame, 
        date_col: Optional[str], 
        validation_strategy: str,
        fold_rows: Optional[int] = None
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generates cross-validation folds based on the specified strategy and data topology.

        With ``rolling_folds=k`` (and a date column, and not a holdout) the folds are rolling origins instead: the last ``k * fold_rows`` dates of the
        pool are cut into ``k`` blocks of ``fold_rows`` dates (the horizon), each validated on a model trained on everything before it.
        """
        if validation_strategy == 'holdout':
            return [(df_fit.copy(), df_dev.copy())]

        df_pool = pd.concat([df_fit, df_dev])
        if date_col and date_col in df_pool.columns:
            df_pool = df_pool.sort_values(date_col)
            if self.rolling_folds:
                rolling = rolling_origin_folds(df_pool, date_col, self.rolling_folds, fold_rows or 1)
                if rolling:
                    return rolling

        is_timeseries = False
        step_size = None

        if validation_strategy in ['auto', 'expanding_window'] and date_col and date_col in df_pool.columns:
            try:
                date_series = pd.to_datetime(df_pool[date_col])
                deltas = date_series.diff().dropna()
                if not deltas.empty:
                    median_delta = deltas.median()
                    if median_delta >= pd.Timedelta(days=28):
                        step_size = 12
                    elif median_delta >= pd.Timedelta(days=1):
                        step_size = 30
                    else:
                        step_size = max(1, len(df_pool) // 10)
                    is_timeseries = True
            except Exception as exc:
                log_failure(logger, "Date and frequency detection of the validation split (falls back to a non-time-series split)", exc)

        if validation_strategy == 'expanding_window' or is_timeseries:
            if step_size is None:
                step_size = max(1, len(df_pool) // 5)
            
            folds = []
            n_samples = len(df_pool)
            if self.max_cv_folds:   # a long daily or monthly series would otherwise have a fold every 30 rows (12 for months)
                step_size = max(step_size, -(-(n_samples - max(step_size * 2, n_samples // 3)) // self.max_cv_folds))
            initial_train_size = max(step_size * 2, n_samples // 3)
            
            current_end = initial_train_size
            while current_end + step_size <= n_samples:
                train_fold = df_pool.iloc[:current_end].copy()
                val_fold = df_pool.iloc[current_end:current_end + step_size].copy()
                folds.append((train_fold, val_fold))
                current_end += step_size
            
            if current_end < n_samples and len(folds) > 0:
                train_fold = df_pool.iloc[:current_end].copy()
                val_fold = df_pool.iloc[current_end:].copy()
                folds.append((train_fold, val_fold))
                
            if not folds:
                folds = [(df_fit.copy(), df_dev.copy())]
            return folds
            
        folds = []
        indices = np.random.RandomState(seed=42).permutation(df_pool.index)
        k = 3 if (self.small_sample and len(indices) < 30) else 5   # a fold of a handful of rows is no ground for a choice
        fold_size = max(1, len(indices) // k)
        
        for i in range(k):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < k - 1 else len(indices)
            val_idx = indices[start_idx:end_idx]
            train_idx = np.setdiff1d(indices, val_idx)
            
            train_fold = df_pool.loc[train_idx].copy()
            val_fold = df_pool.loc[val_idx].copy()
            folds.append((train_fold, val_fold))
            
        return folds
#: </data_manager_cv>

#: <data_manager_structure>
    def _declare_structure_columns(self, frame: pd.DataFrame, ctx: PipelineContext, date_col: Optional[str], group_col: Optional[str]) -> None:
        """Finds the exposure, the coordinate pairs and the identifiers, and tells the feature engineer to leave them alone."""
        reserved = {ctx.target, date_col, group_col}
        ctx.offset_col = self.offset_col
        if self.offset_col:
            if self.offset_col not in frame.columns:
                raise ValueError(f"offset_col '{self.offset_col}' is not a column of the data.")
            self.engineer.no_ratio.add(self.offset_col)
            self.engineer.do_not_smooth_features.add(self.offset_col)
        features = [c for c in frame.columns if c not in reserved and c != self.offset_col]
        if self.find_spatial_pairs:
            ctx.spatial_pairs = spatial_pairs(features)
            for pair in ctx.spatial_pairs:
                self.engineer.no_ratio.update(pair)
            if ctx.spatial_pairs:
                logger.info("Spatial coordinate pairs found: %s.", ", ".join(f"{a}/{b}" for a, b in ctx.spatial_pairs))
        if self.find_random_effects:
            ctx.random_effect_columns = random_effect_columns(frame[features])
            for column in ctx.random_effect_columns:
                self.engineer.no_ratio.add(column)
                self.engineer.do_not_smooth_features.add(column)
            if ctx.random_effect_columns:
                logger.info("Identifier columns with many levels (random effects): %s.", ", ".join(ctx.random_effect_columns))
#: </data_manager_structure>

#: <data_manager_calendar>
    def _add_calendar_columns(self, df_aug: pd.DataFrame, ctx: PipelineContext) -> pd.DataFrame:
        """Adds the calendar and holiday columns (after the engineering, so they are never smoothed) and offers them to the search."""
        if not (self.calendar_features or self.country):
            return df_aug
        if not ctx.date_col or ctx.date_col not in df_aug.columns:
            raise ValueError("calendar_features and country need a date column: pass date_col=.")
        spd = steps_per_day(df_aug[ctx.date_col])
        names = calendar_names(spd) if self.calendar_features else []
        country = self.country
        if self.calendar_guards:
            names, dropped = guarded_calendar_names(names, df_aug[ctx.date_col], spd)
            for name, reason in dropped.items():
                logger.info("Calendar guard: '%s' is not offered: %s.", name, reason)
            if country and (spd is None or spd < MIN_STEPS_PER_DAY_FOR_HOLIDAYS):
                logger.info("Calendar guard: the holidays of '%s' are not offered at a step coarser than a day.", country)
                country = None
        ctx.calendar = {"names": names, "country": country, "subdivision": self.subdivision, "steps_per_day": spd,
                        "extra_flags": self.calendar_guards}
        df_aug = df_aug.copy()
        features = list(ctx.formula_config["features"])
        among = [c for c in df_aug.columns if any(c == f or c.startswith(f"{f}_") for f in features)] if features else None
        added = add_calendar(df_aug, ctx.date_col, ctx.calendar, skip_equivalent=True, among=among)
        group_col = ctx.group_col
        if group_col and group_col in df_aug.columns and added:
            # A calendar column that never changes inside a group (the day of the week of a group per weekday, the time of day of a group per
            # period) carries nothing a per-group model can learn, and makes its fit singular: it is left out.
            constant = [name for name in added if int(df_aug.groupby(group_col)[name].nunique().max()) <= 1]
            if constant:
                df_aug = df_aug.drop(columns=constant)
                added = [name for name in added if name not in constant]
                ctx.calendar["keep"] = list(added)
                logger.info("Calendar columns constant inside each group are not added: %s.", ", ".join(constant))
        for name in added:
            if name not in ctx.formula_config["features"]:
                ctx.formula_config["features"].append(name)
        ctx.calendar["columns"] = added
        periods = {"toy": (0, 1), "tod": (0, 1), "dow": "auto"}
        ctx.formula_config["calendar_columns"] = {name: periods[name] for name in added if name in periods}
        return df_aug
#: </data_manager_calendar>

#: <data_manager_crisis>
    def _add_crisis_columns(self, df_aug: pd.DataFrame, ctx: PipelineContext) -> pd.DataFrame:
        """Adds the indicator and the age of each dated event that overlaps the data, and offers them to the search."""
        if not self.crisis_events:
            return df_aug
        if not ctx.date_col or ctx.date_col not in df_aug.columns:
            raise ValueError("crisis_calendar needs a date column: pass date_col=.")
        events = active_events(self.crisis_events, df_aug[ctx.date_col])
        if not events:
            logger.info("Crisis calendar: no event overlaps the dates of the data.")
            return df_aug
        df_aug = df_aug.copy()
        added = add_crisis(df_aug, ctx.date_col, events)
        for name in added:
            if name not in ctx.formula_config["features"]:
                ctx.formula_config["features"].append(name)
        ctx.crisis = {"events": events, "columns": added}
        logger.info("Crisis calendar: %d event(s) overlap the data: %s.", len(events), ", ".join(e["name"] for e in events))
        return df_aug
#: </data_manager_crisis>

#: <data_manager_floor>
    def _add_floor_column(self, df_aug: pd.DataFrame, ctx: PipelineContext) -> pd.DataFrame:
        """Adds the position of each row inside its natural season, the column of the pooled seasonal floor (not offered to the search)."""
        if not self.pooled_floor:
            return df_aug
        if not ctx.date_col or ctx.date_col not in df_aug.columns:
            raise ValueError("pooled_floor needs a date column: pass date_col=.")
        spd = steps_per_day(df_aug[ctx.date_col])
        positions, count = season_position(df_aug[ctx.date_col], spd)
        ctx.floor = {"steps_per_day": spd, "count": count}
        df_aug = df_aug.copy()
        df_aug[FLOOR_COLUMN] = positions
        return df_aug
#: </data_manager_floor>

#: <data_manager_regimes>
    def _add_regime_column(self, df_aug: pd.DataFrame, ctx: PipelineContext) -> pd.DataFrame:
        """Detects the regimes of the training target and adds the ``regime`` column (a category for the search); nothing when none is found."""
        if not self.regimes:
            return df_aug
        if not ctx.date_col or ctx.date_col not in df_aug.columns:
            raise ValueError("regimes needs a date column: pass date_col=.")
        rows = df_aug[df_aug[ctx.target].notna()]
        season = natural_season(ctx.metadata.get("delta_t", "1D"))
        fitted = fit_regimes(rows, ctx.target, ctx.date_col, ctx.group_col, season)
        if not any(fitted["breaks"].values()):
            logger.info("Regimes: no level shift found in the training target.")
            return df_aug
        df_aug = df_aug.copy()
        add_regime_column(df_aug, ctx.date_col, ctx.group_col, fitted)
        ctx.regime = fitted
        if "regime" not in ctx.formula_config["features"]:
            ctx.formula_config["features"].append("regime")
        ctx.formula_config["categorical_columns"] = list(ctx.formula_config.get("categorical_columns", [])) + ["regime"]
        logger.info("Regimes: %d break(s) found in the training target.", sum(len(v) for v in fitted["breaks"].values()))
        return df_aug
#: </data_manager_regimes>

#: <data_manager_transform>
    def transform_test_data(self, df_test: pd.DataFrame, ctx: PipelineContext) -> pd.DataFrame:
        """
        Transforms unseen test data during the predict phase, utilizing the historical tail.
        """
        if hasattr(ctx, "external_mandatory_features") and ctx.external_mandatory_features:
            missing_ext = set(ctx.external_mandatory_features) - set(df_test.columns)
            if missing_ext:
                raise ValueError(
                    f"Test data is missing required external mandatory features: {sorted(missing_ext)}"
                )

        df_combined = df_test.copy()
        if getattr(ctx, "sentinel_cleaner", None) is not None:
            df_combined = ctx.sentinel_cleaner.apply(df_combined)
        
        if ctx.historical_tail is not None and ctx.date_col and ctx.date_col in df_test.columns:
            test_dates = df_test[ctx.date_col]
            if pd.api.types.is_datetime64_any_dtype(ctx.historical_tail[ctx.date_col]):
                test_dates = pd.to_datetime(test_dates, errors="coerce")   # a frame loaded from a file has string dates; the tail has Timestamps
            if test_dates.min() > ctx.historical_tail[ctx.date_col].max():
                df_combined = pd.concat([ctx.historical_tail, df_test.assign(**{ctx.date_col: test_dates})], ignore_index=True)

        if ctx.lags:
            for d in ctx.lags: 
                if ctx.group_col and ctx.group_col in df_combined.columns:
                    df_combined[f"{ctx.target}_lag_{d}"] = df_combined.groupby(ctx.group_col)[ctx.target].shift(d)
                else: 
                    df_combined[f"{ctx.target}_lag_{d}"] = df_combined[ctx.target].shift(d)

        add_past_covariate_lags(df_combined, ctx.past_covariates, ctx.group_col)
        if ctx.anchor_season:
            add_anchor_lag(df_combined, ctx.target, ctx.anchor_season, ctx.group_col)

        df_clean = self.profiler.transform(df_combined, ctx.formula_config, ctx.date_col) if hasattr(self.profiler, 'transform') else df_combined
        df_aug = self.engineer.engineer_features(df_clean, ctx.formula_config, ctx.metadata, ctx.date_col)
        df_aug = keep_forecast_targets_missing(df_aug, df_combined, ctx)
        if ctx.calendar:
            add_calendar(df_aug, ctx.date_col, ctx.calendar)
        if ctx.crisis:
            add_crisis(df_aug, ctx.date_col, ctx.crisis["events"], ctx.crisis["columns"])
        if ctx.regime:
            add_regime_column(df_aug, ctx.date_col, ctx.group_col, ctx.regime)
        if ctx.floor:
            df_aug[FLOOR_COLUMN] = season_position(df_aug[ctx.date_col], ctx.floor["steps_per_day"])[0]

        # Tz-safe left-merge to perfectly realign predictions with df_test's original row order.
        date_col, group_col = ctx.date_col, ctx.group_col
        try:
            if date_col and date_col in df_aug.columns and date_col in df_test.columns:
                gcol = group_col if (group_col and group_col in df_aug.columns and group_col in df_test.columns) else None
                freq = ctx.metadata.get("delta_t", "1D")
                keys = ["__kdate__"] + ([gcol] if gcol else [])

                a = df_aug.copy()
                a["__kdate__"] = pd.to_datetime(a[date_col])
                left = df_test[[date_col] + ([gcol] if gcol else [])].copy()
                left["__kdate__"] = pd.to_datetime(left[date_col])
                try:
                    a["__kdate__"] = a["__kdate__"].dt.floor(freq)
                    left["__kdate__"] = left["__kdate__"].dt.floor(freq)
                except (ValueError, TypeError):
                    pass

                a = a.drop(columns=[date_col]).drop_duplicates(subset=keys, keep="last")
                left["__ord__"] = range(len(left))
                merged = left.merge(a, on=keys, how="left").sort_values("__ord__")
                merged = merged.drop(columns=["__kdate__", "__ord__"]).reset_index(drop=True)

                if len(merged) == len(df_test) and ctx.target in merged.columns and merged[ctx.target].notna().all():
                    return merged, df_aug
        except Exception as exc:
            log_failure(logger, "Alignment of the test frame on the augmented frame (falls back to the tail rows)", exc)

        return df_aug.iloc[-len(df_test):].copy(), df_aug
#: </data_manager_transform>
#: </data_manager_class>