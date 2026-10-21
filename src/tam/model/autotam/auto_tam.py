# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-FileContributor: Amaury Durand
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
AutoTAM Orchestrator (TAM AutoML Layer).

This module manages the end-to-end automated machine learning flow for time-series forecasting.
It acts as the central execution engine, coordinating data topology mapping, evolutionary 
search, state-space expansions (Kalman/Adaptive), and Minimax aggregation (OPERA).
"""

#: <auto_tam_imports>
import logging
import time
import pandas as pd
import numpy as np
import pathlib
import re
from typing import Any, Tuple, Optional, List, Dict, Sequence, Union

from .pipeline.data_manager import DataManager
from .pipeline.base_discoverer import BaseDiscoverer
from .pipeline.expert_expander import ExpertExpander
from .pipeline.ensemble_selector import EnsembleSelector
from .evaluation.autotam_report_generator import generate_autotam_report, print_model_recipe
from .evolution_reporter import EvolutionReporter
from tam.model.kalman import KalmanTAM
from tam.model.adaptative import AdaptiveTAM
from tam.model.opera import OperaTAM
from tam.common.exceptions import collect_extrapolation
from tam.model.autotam.failures import log_failure
from tam.common.utils import parse_formula_to_terms
from tam.model.autotam.anchor import ANCHOR_ISLAND, anchor_formula, natural_season
from tam.model.autotam.floor import FLOOR_COLUMN, FLOOR_ISLAND, floor_formula
from tam.model.autotam.group_search import GROUP_COLUMN, add_group, covariates_in_cycles, derived_group, formula_in_cycles, sample_groups, to_cycles
from tam.model.autotam.alignment import aligned_values
from tam.model.autotam.expert_ids import experts_table, identifiers
from tam.model.autotam.inactivity import InactivityGate, drop_leading_zeros
from tam.model.autotam.covariate_ranking import rank_covariates
from tam.model.autotam.tokens import formula_columns, term_columns
from tam.model.autotam.structure_columns import with_offset
from tam.model.autotam.series_guard import GUARD_ISLANDS, apply_guard, guarded_series
from tam.model.autotam.external_experts import look_up, normalise_external
from tam.model.autotam.seeding import seeded_method
from tam.model.autotam.differencing import add_back, difference_test, difference_training
from tam.model.autotam.target_profile import profile_target
from tam.model.autotam.diagnostics import feature_diagnostics, residual_advice, residual_diagnostics, stationarity_verdict, target_diagnostics
from tam.model.autotam.families import candidate_losses, check_family_mode
from tam.model.autotam.spectral_profile import spectral_profile
from tam.model.autotam.calendar_features import calendar_seed_formulas
from tam.model.autotam.search_preset import (
    FAST_GENERATIONS, FAST_ISLANDS, FAST_MAX_CV_FOLDS, FAST_N_EXPERTS, FAST_POP_SIZE, FULL_GENERATIONS, check_search_mode)
logger = logging.getLogger(__name__)
#: </auto_tam_imports>

#: <auto_tam_forecast_rows>
def split_forecast_rows(df: pd.DataFrame, target: str, group_col: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits a time-ordered frame into the rows whose target is observed and the forecast rows.

    A forecast row has no target and no observed target after it (inside its group): these are the rows after the last observation. A row
    with a missing target followed by an observation is a gap, neither observed nor a forecast row.

    Args:
        df: Frame in time order, with the target column.
        target: Name of the target column.
        group_col: Name of the group column, or ``None``.

    Returns:
        The observed rows and the forecast rows.
    """
    observed = df[target].notna()
    reversed_observed = observed.iloc[::-1]
    if group_col and group_col in df.columns:
        later = reversed_observed.groupby(df[group_col].iloc[::-1]).cummax().iloc[::-1]
    else:
        later = reversed_observed.cummax().iloc[::-1]
    return df[observed], df[~later.astype(bool)]
#: </auto_tam_forecast_rows>

#: <auto_tam_finite_mix>
def finite_weighted_mix(frame: pd.DataFrame, weights: Dict[str, float]) -> Tuple[pd.Series, int]:
    """Weighted mean of the experts of an ensemble, over the experts that are finite on each row.

    On a row where every member is finite the result is the plain weighted mean (the weights divided by their sum, as before). On a row where a
    member is missing it is dropped and the weights of the others are renormalised, so one expert that cannot forecast a row does not make the
    ensemble missing there. A row where no member is finite stays missing.

    Args:
        frame: Predictions, one column per expert.
        weights: Weight of each member, by column name; every member must be a column of ``frame``.

    Returns:
        The mixed series, and the number of (row, expert) values that were left out.
    """
    total = sum(weights.values())
    plain = sum(frame[name] * (weight / total) for name, weight in weights.items())
    values = frame[list(weights)]
    finite = values.notna()
    complete = finite.all(axis=1)
    if complete.all():
        return plain, 0
    weight_row = pd.Series(weights)
    numerator = (values.fillna(0.0) * weight_row).sum(axis=1)
    denominator = (finite * weight_row).sum(axis=1)
    mixed = (numerator / denominator).where(denominator > 0)
    return plain.where(complete, mixed), int((~finite).to_numpy().sum())
#: </auto_tam_finite_mix>

#: <auto_tam_class>
class AutoTAM:
    """The Clean AutoTAM Director.

    Attributes:
        FINAL_MODEL_COLUMN: Name of the column of ``predict()`` that holds the final model, the Apex ensemble.
        FINAL_MODEL_COLUMN_ONLINE: Name of the same model in the output of ``predict_online()``.
        final_model_column_: ``FINAL_MODEL_COLUMN`` once ``fit()`` has kept experts for the Apex ensemble, else ``None``.
    """

    FINAL_MODEL_COLUMN = "AutoTAM_Apex_Ensemble"
    FINAL_MODEL_COLUMN_ONLINE = "AutoTAM_Apex_Online"

#: <auto_tam_init>
    def __init__(
        self, 
        formula: str, 
        lags: Optional[List[int]] = None, 
        mandatory_variables: Optional[List[str]] = None,
        n_experts: Optional[int] = None, 
        pop_size: Optional[int] = None, 
        use_opera: bool = True, 
        eta: float = 0.1, 
        complexity_penalty: float = 1.0,
        export_dir: str = "AutoTAM_exports", 
        extrapolation_guards: bool = False,
        seasonal_anchor: Optional[bool] = None,
        season: Optional[int] = None,
        replace_sentinels: bool = False,
        seed: Optional[int] = None,
        horizon: Optional[int] = None,
        search: str = "full",
        calendar_features: bool = False,
        country: Optional[str] = None,
        subdivision: Optional[str] = None,
        family: str = "gaussian",
        search_groups: Optional[int] = None,
        search_sample: str = "level",
        diagnostics: bool = False,
        regimes: bool = False,
        apex_quality_ratio: Optional[float] = 1.5,
        frozen_weights: str = "snapshot",
        extended_search: bool = False,
        seed_floor: bool = False,
        difference: Union[None, int, str] = None,
        dynamic_top: Optional[int] = None,
        rolling_folds: Optional[int] = None,
        pooled_floor: bool = False,
        group_search: bool = False,
        group_search_by: str = "validation",
        online_variants: bool = False,
        crossover: bool = False,
        native_quantiles: bool = False,
        crisis_calendar: Any = None,
        guided_search: bool = False,
        bounded_rewards: bool = False,
        apex_prune: Optional[float] = None,
        series_guard: bool = False,
        max_cv_folds: Optional[int] = None,
        clean_sentinels: bool = False,
        calendar_guards: bool = False,
        smooth_monotonic: bool = False,
        cross_sectional: bool = False,
        tensor_margin_penalty: bool = False,
        inactivity_gate: bool = False,
        closed_when: Optional[Dict[str, float]] = None,
        rank_covariates: bool = False,
        max_covariates: Optional[int] = None,
        physical_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        family_search: bool = False,
        offset_col: Optional[str] = None,
        spatial_pairs: bool = False,
        random_effects: bool = False,
        strong_regularization: bool = False,
        small_sample: Union[bool, str] = False,
        autofit_criterion: str = "gcv",
        transform_experts: bool = False,
        family_adaptive: bool = False,
        extended_effects: bool = False,
        **kwargs
    ):
        """
        Initializes the AutoTAM director.

        Warnings and progress messages go through ``logging`` (logger ``tam.model.autotam``): see the progress with
        ``logging.basicConfig(level=logging.INFO)``, silence it with ``logging.getLogger('tam.model.autotam').setLevel(logging.ERROR)``. The
        reports (``summary()``, ``print_performance_board()``, ``print_model_recipe()``) still print.

        Args:
            formula (str): High-level AutoTAM formula with embedded mandatory terms and a
                pipeline macro (e.g., 'load ~ s(temp, k=10) + AutoPipe(temp, humidity)').
            lags (Optional[List[int]]): Explicit lag orders to evaluate during feature engineering.
            search (str): ``"full"`` (default) is the search as it always was. ``"fast"`` is a preset for series that do not need the whole
                search: the linear, spline, Fourier and Chebyshev islands only (no neural, RBF, tree, wavelet or mixed islands), a population of
                16 over 8 generations instead of 64 over 20, 8 experts kept, three Kalman noise rates and two adaptive windows, and the seasonal
                anchor on. ``pop_size``, ``n_experts`` and ``seasonal_anchor`` given by the caller win over the preset. It is a screening preset,
                about twice as fast: it leaves out the islands that hourly prices need, so it is not meant for final forecasts.
            calendar_features (bool): Opt-in. Derives calendar positions from the date column (``toy`` time of year and ``tod`` time of day as
                fractions in [0, 1), ``dow`` day of week), offers them to the search as cyclic terms only (never linear), runs the spectral profile
                of the training target and seeds the cycles it finds (``calendar_profile_``). Needs ``date_col``. A position the frame already
                has under another name (an hour column, a day of week) is not added again. It is not a default: it was measured worse on
                hourly electricity prices, where the caller already gives the calendar columns.
            country (Optional[str]): Opt-in. Country code of the public holidays (``'FR'``): adds ``holiday``, ``days_to_holiday``,
                ``days_since_holiday`` and ``bridge_day``. Needs the ``holidays`` package (``pip install tam-ml[calendar]``) and ``date_col``. A holiday flag
                the frame already has is not added again. Not a default: it was measured worse on hourly prices that already carry holiday columns.
            subdivision (Optional[str]): Region of the holiday calendar (``'IDF'``), with ``country``.
            family (str): ``"gaussian"`` (default) keeps the identity link and the L2 loss. ``"auto"`` also fits each champion formula under the
                losses the profile of the training target allows (Gamma for a positive skewed target, Tweedie for a non-negative target with zeros,
                Poisson and negative binomial for counts, Huber for heavy tails), scored in the response scale, so the family competes with the
                Gaussian expert in the leagues. A log link is never proposed for a target with zeros or negative values.
            family_search (bool): Opt-in, with ``family="auto"`` and a target the profile gives a family (binomial, Gamma, Poisson, negative
                binomial, Tweedie). The evolutionary search itself fits its candidates under the first family of the profile (a log or logit link)
                instead of the identity link: on a multiplicative process the identity link needs interaction terms that an additive model on the
                link scale does not, and the champions are then fitted again under the family. ``False`` (default) searches on the identity link.
            offset_col (Optional[str]): Opt-in. The column of a known exposure or effort (the log of an area, of a volume, of the observation time),
                with a fixed coefficient of one. Every candidate the search fits and every expert carries ``offset(<col>)``; the column is not a
                feature (no ratio, no curve, no tensor product with it). The engine refuses an offset in the Kalman and adaptive stages, so
                they are not built. The column must be in the frames given to ``predict()``.
            spatial_pairs (bool): Opt-in. Pairs of coordinate columns among the features (``lon``/``lat``, ``x``/``y``, ``e``/``n``...) are found:
                no ratio is made from them, and two seed experts are added, a thin-plate surface ``rbf(a, others='b', kernel='tps')`` and a tensor
                product ``te(s(a), s(b))``, so a spatial surface competes from the start. ``spatial_pairs_`` lists the pairs.
            random_effects (bool): Opt-in. Identifier columns with many levels (text with 16 to 500 levels, or whole numbers with 16 to 250 that
                repeat and are named like an identifier or are perfectly balanced) get a ridge-shrunk ``c(id, ap=-2)`` as their only effect, are
                left out of ratios and smoothing, and are offered as the partner of a factor smooth ``te(s(x), c(id))``.
                ``random_effect_columns_`` lists them.
            strong_regularization (bool): Opt-in. The smoothing penalties searched for the curves (``s``, ``f``, ``p``, ``w``) include the strong
                ones (``ap`` of -1 and 0, about 2 degrees of freedom, so a small or noisy sample can keep a smooth instead of falling back on a line)
                and, from 100 rows, the weak one (-6); the margins of a tensor product take -3, -1 or 0; and a champion whose cross-validated R2
                is above 0.92 is also fitted with more knots and the locally adaptive penalty ``m_pen=4`` (the experts ``..._HighSNR_...``).
                ``False`` (default) is the search space as it was.
            small_sample (Union[bool, str]): Opt-in, for data with no date column and few rows. ``True``, or ``"auto"`` when there is no date column
                and fewer than 150 rows: the candidates are scored on 5 folds (3 under 30 rows) of the training rows and their error is multiplied
                by the corrected-AIC factor ``1 + 2k / (n - k - 1)`` (k the coefficients), a formula has at most ``n // 8`` terms, and a heavily
                regularised linear model of the first three features is an expert the final model cannot be worse than (the seed floor).
                ``is_small_sample_`` says whether it was on. ``False`` (default) is the search as it was.
            autofit_criterion (str): ``"gcv"`` (default) or ``"reml"``: what the ``auto_fit`` experts (the Gaussian ones and, with ``family="auto"``,
                the family ones) minimise to choose their smoothing parameters. REML (Laplace restricted marginal likelihood) is less prone to
                undersmoothing than GCV when the residuals are autocorrelated.
            transform_experts (bool): Opt-in. Static experts on a transformed target, from the profile of the target (``log``, ``asinh``, ``logit``,
                ``log1p``), forecast back on the scale of the target.
            family_adaptive (bool): Opt-in, with ``family="auto"`` and the adaptive expansion. Adaptive experts on the family expert that beats the
                Gaussian static expert on the folds (windows fitted on the link scale, so the forecast keeps the support of the family).
            extended_effects (bool): Opt-in. Parameters of the effects that the search left fixed are searched too: the thin-plate kernel of
                ``rbf``, the locally adaptive penalty ``m_pen`` of ``s``, ``orthogonalize`` for ``n`` and ``t``, the ``ordinal`` topology and
                ``unseen`` of ``c``, and ``mode='per_level'`` of a tensor product with a factor.
            search_groups (Optional[int]): For data with thousands of groups: ``fit()`` searches on this many groups, taken evenly spaced by the
                level of their target, then fits the architectures found on every group (the way ``architectures=`` does). ``None`` (default)
                searches on all groups; it has no effect without a ``group_col`` or when there are not more groups than this number. The groups
                searched are in ``search_sample_``.
            search_sample (str): How ``search_groups`` picks them: ``"level"`` (default) evenly spaced in the order of the level of the target;
                ``"level_volatility"`` half of them in that order and half evenly spaced in the order of the coefficient of variation, so steady and
                erratic series are both searched.
            diagnostics (bool): Opt-in. ``fit()`` also runs statistical diagnostics: of the training target (unit root and stationarity, seasonal
                strength, spread, tails, outliers, demand class, break, autocorrelation) in ``target_diagnostics_``, and of the residuals of the
                final model on the validation rows (structure left, volatility clustering, bias) in ``residual_profile_``. They are hints for
                the user, they do not change the search.
            regimes (bool): Opt-in. Detects level shifts in the training target (a lockdown, a crisis, a change of process) by binary segmentation
                of its level with the seasonal profile removed, and offers the regime number to the search as a category. Only the data of the task
                are used; a regime lasts at least four seasons, and a forecast row carries the regime in force at the end of the training rows, or the
                previous one while the last has lasted fewer than eight seasons. Needs ``date_col``. Not a default: it was measured worse on hourly
                electricity prices.
            apex_quality_ratio (Optional[float]): A dynamic expert joins the Apex pool only if its validation error is at most this multiple of the
                best dynamic expert's (default 1.5, the search as it was). ``None`` admits the whole pool.
            frozen_weights (str): What the frozen ``predict()`` takes as the weight of an expert: ``"snapshot"`` (default) the last online weight
                of the validation rows, ``"mean_7"`` / ``"mean_30"`` the mean over the last 7 or 30 days, ``"full"`` the mean over the whole
                validation part.
            extended_search (bool): Opt-in wider search space: more spline knots and RBF centres (up to a twentieth of the rows, at most 50), a
                wider PID window grid, constant features left out, and a mutation that may add a second basis on a feature the formula already uses. ``False``
                (default) is the search as it was. With ``search="fast"`` it improved the forecast of hourly prices and of weekly orders for about 30 % more
                time; it was not checked with ``search="full"``, so it stays off by default.
            seed_floor (bool): Opt-in, for runs with ``seed_formulas``. The seed experts always join the Apex pool, whatever ``apex_quality_ratio``,
                and the frozen final model is never worse on the validation rows than the best of them: when it is, that seed expert is the final
                model (logged). The validation rows are the ones the search saw, so the floor is a guard, not an unbiased estimate.
            difference (Union[None, int, str]): Opt-in. ``"select"`` fits a level model and a differenced model on the earlier rows, forecasts the last
                three horizons with each, and keeps the one with the smaller error in the level scale (``difference_scores_``); it costs two more fits.
                ``h`` (a whole number of steps) makes the model learn the increments ``y_t - y_{t-h}`` and
                forecast the level by adding them back step by step; ``"auto"`` does it, with ``h`` the horizon (the smallest lag, else 1), when the
                profile of the training target says it is cumulative or monotone, or the Dickey-Fuller and KPSS tests find a unit root. For a cumulative series, a unit root or quarterly and yearly data,
                where the level has no stable mean. Needs ``date_col``. ``predict()`` and ``predict_quantiles()`` return levels; ``predict_online()``
                is not available. The lags and the anchor are on the increments. ``None`` (default) is the search as it was.
            pooled_floor (bool): Opt-in. Adds to the pool a seasonal profile (one coefficient per position of the natural season of the sampling
                step: the hour of the day, the day of the week, the week of the year, the month or the quarter), fitted for all the series at once.
                It always joins the Apex pool and the frozen final model is never worse than it on the validation rows (logged when it is the
                final model). The benchmark of seasonal series, as a guard. Needs ``date_col`` and a step with a natural season.
            guided_search (bool): Opt-in. The diagnostics guide the search, as hints and never as constraints: it switches on ``diagnostics``,
                ``difference="auto"`` (the increments are modelled when the target is cumulative, monotone or has a unit root) and ``family="auto"``
                (the count and positive-target families are tried where the profile of the target asks for them) unless you set them, screens the
                features (Spearman correlation with the target at the same step and at the lag of the horizon; the weak ones are listed), and
                writes what the residual diagnostics suggest. All of it is in ``guided_hints_``. The search itself is the one of those options.
            bounded_rewards (bool): Opt-in. The search draws the islands with a bandit whose reward is ``1 / scaled error``, which reaches 10 to
                100 for a good model against an exploration term below 1, so the best island is drawn almost every time. ``True`` uses the bounded
                reward ``1 / (1 + scaled error)`` in (0, 1] and a smaller exploration constant (0.1), so the islands keep being tried in proportion
                to their results. ``False`` (default) is the search as it was.
            apex_prune (Optional[float]): Opt-in. A member of the Apex pool whose online weight stays below this value (for example 0.02) on the
                last 10 updates of the validation rows is dropped, the aggregation is run again on the rest, and an expert that no ensemble uses any
                more is no longer predicted, which cuts the cost of ``fit`` and ``predict``. The experts of ``seed_floor`` / ``pooled_floor`` are never
                dropped; the names dropped are in ``apex_pruned_``. ``None`` (default) keeps the pool as it was.
            max_cv_folds (Optional[int]): At most this many expanding-window folds in the usual validation of a daily or monthly series (the step
                of the folds grows to fit). Without it a series of six years of days has a fold every 30 rows, 48 folds to score every candidate on.
                ``None`` (default) keeps the folds as they were; ``search="fast"`` takes 5 unless you give it.
            smooth_monotonic (bool): Opt-in. A feature whose values only go one way is no longer forced to a linear term. A time counter or a trend
                of a series (uniform steps, with a date column) may also take a low-degree smooth (``p`` of degree 2 or 3, ``s`` with 4 knots); a
                predictor that is merely sorted (a table ordered by x, a column with irregular steps, or no date column) is profiled like any
                other and may take any smooth. The lags of the target stay linear. ``False`` (default) is the search space as it was.
            cross_sectional (bool): Opt-in, for data with no time order (``date_col=None``). ``df_train`` is shuffled (seed 42) before it is split in
                fit, development and validation (a split in row order validates on one end of the covariates), the Kalman and adaptive experts are
                not built unless ``expansions`` says so, and a numeric column is a category only if its values are whole numbers and it has no
                more distinct values than half the rows. ``False`` (default) is as it was.
            tensor_margin_penalty (bool): Opt-in. A ``te(...)`` term without an ``ap=`` of its own carries the engine default 1e-9 as the weight of
                the product, and the engine folds it into the penalty of every margin: the margins the search wrote with ``ap=-3`` are fitted at
                1e-12. ``True`` writes ``ap=0.0`` on the tensor products the search fits and returns, so each margin keeps the penalty it was
                given. ``False`` (default) is the search as it was.
            calendar_guards (bool): Opt-in, with ``calendar_features`` and/or ``country``. The time of year (``toy``) is offered only with at least
                two years of dates and a step finer than a year (a shorter history makes the model memorise the one-off events of its single year;
                at a yearly step it is a constant); the holidays of ``country`` only at a step of a day or finer; and the holiday columns also
                get ``day_before_holiday``, ``day_after_holiday`` (the surge before closing, the catch-up after) and ``year_end_week`` (24
                December to 2 January). What is left out is logged. ``False`` (default) keeps the calendar columns as they were.
            inactivity_gate (bool): Opt-in, for series that may not be running. The rows before the first non-zero target of each group are left out
                of the fit (a shop opened last month is not estimated from years of zeros); a group whose training target is all zero (not opened),
                the positions of the season that are zero in 95 % of the training values (the night of a solar plant, for sub-daily data), and the
                rows declared closed by ``closed_when`` are forecast as zero, point and quantiles, in every column of ``predict()``,
                ``predict_online()`` and ``predict_quantiles()``. ``inactive_groups_`` lists the groups whose last 12 training values are zero
                (reported, not clamped). ``False`` (default) leaves the forecasts as they were.
            closed_when (Optional[Dict[str, float]]): With ``inactivity_gate``, known-future columns that say the series is closed: ``{"Open": 0,
                "shops_closed": 1}`` forecasts zero on the rows where ``Open`` is 0 or ``shops_closed`` is 1.
            rank_covariates (bool): Opt-in. The covariates are ranked before the search by the absolute Spearman correlation with the target (the
                target standardised inside each group; a past-only covariate is its lag column, so its correlation is the one at the lag of the
                horizon), and the search draws the covariates of a candidate with a probability that grows with that score instead of uniformly.
                ``covariate_ranking_`` is the table (score, and the higher ranked covariate it repeats above 0.95). The calendar, crisis and regime
                columns, the lags of the target and the anchor are not ranked; the mandatory variables are never left out.
            max_covariates (Optional[int]): With the ranking (it switches it on), keep only the ``max_covariates`` best covariates that do not
                repeat a better one (the features of the search space, the engineered ones - ratios, rolling means - included); the others are not
                offered to the search. ``None`` (default) keeps them all.
            clean_sentinels (bool): Opt-in. Before the lags are built, the missing-data codes and impossible readings of the target and of the
                numeric features are replaced by the median of the same position of the season (hour of the day, day of the week, month) of the
                training rows, in the same group: round negative constants that repeat (-9999, -999, -200 ...), the rare negatives of a target that
                is never negative (at most 0.5 %), readings outside the physical range of a temperature in degrees Celsius (-60 to 60) or of a wind
                direction (0 to 360). The same rules are applied to the rows to forecast; only the cells they flag are filled, a missing target stays
                missing. ``sentinel_report_`` lists them. Unlike ``replace_sentinels`` (the target only, missing values left as they are) it also
                cleans the features and fills what it masks.
            physical_bounds (Optional[Dict[str, Tuple[float, float]]]): With ``clean_sentinels``, the range of a column, ``{"Etmp": (-60, 60)}``,
                beyond which a reading is not one (a column that is not named like a temperature or a direction has no range otherwise).
            series_guard (bool): Opt-in, for panels of series (a ``group_col``, or a derived group) and for a single series. After the fit, each
                group whose validation RMSE is smaller for a guard expert than for the final model is forecast by that expert, in the final model's
                column of ``predict()`` and ``predict_online()``. The guard experts are those of the pooled floor (``pooled_floor=True``) and of the
                seasonal anchor (``seasonal_anchor=True``): at least one is needed. ``guarded_series_`` lists the groups. The floors bound the final
                model over all the rows together; this bounds it series by series. ``False`` (default) leaves the final model as it was.
            crisis_calendar (Any): Opt-in dated events as features, known in advance like holidays: ``True`` or ``"all"`` (the presets
                ``covid19``, ``recession_2008``, ``energy_2022``: worldwide windows, approximate), a preset name, or a list of preset names and of
                your own events ``{"name": "strike", "start": "2023-03-07", "end": "2023-04-20"}``. Each event that overlaps the data adds
                ``crisis_<name>`` (1 inside the event) and ``crisis_<name>_age`` (years since its start, capped at 2) to the features the search
                may use; the dates are applied to the rows to forecast as they are. Needs ``date_col``. ``None`` (default) adds nothing.
            native_quantiles (bool): Opt-in. The best champion of each island also gets a distributional expert: a ``StaticTAM`` with the champion
                formula as the location and a scale linear in its numeric features, so its quantiles widen where the data are noisier. Its point
                forecast (the median) joins the pool like any static expert, but not the Apex: the point forecast of the final model is not changed.
                ``predict_quantiles(aggregation="opera")`` aggregates it as one more member, with its own quantiles, next to the members of the final
                model (their point forecast plus validation-error offsets); the pinball loss of each level weights it (``quantile_members_``).
                ``False`` (default) is the pool as it was.
            crossover (bool): Opt-in search operators. A mutation may take one additive branch (a term, or a tensor product) from the champion of
                another island and add it to the champion it mutates (replacing a term when the island is at its limit), and may insert a tensor
                product of a used feature with another feature. Both respect the covariate lock and the eligible effects of the search space.
                ``False`` (default) is the search as it was.
            online_variants (bool): Opt-in. More configurations of the Kalman and adaptive experts of the best champions: two adaptive formulas
                with the static forecast as an input (the lagged residuals plus ``l(Estimated<target>)``, with and without the base effects), Kalman
                experts with an absolute process noise of 1e-6, 1e-5, 1e-4 and 1e-3 next to the rates relative to the observation noise, and an adaptive
                window of 350 periods when it is at most half of the training rows. It adds experts, so it adds time; use it with ``dynamic_top`` on large tasks.
                ``False`` (default) is the search as it was.
            group_search (Union[bool, str]): Opt-in, for a series given without ``group_col``. ``"always"`` fits only the grouped version (the
                period of the day as the group): on the hourly day-ahead load it was better on the hold-out on 3 of 3 seeds and cost about 0.6 of
                the plain fit, while the comparison of ``True`` chose the ungrouped fit on 2 of 3 seeds, wrongly (validation error does not rank
                the two). ``True`` is as follows. ``fit()`` also fits the model with a group column derived from the
                dates (one group per period of the day on sub-daily data, one step being one day; one per day of the week on a daily series) and keeps
                the version with the smaller error on the development and validation dates both have (``group_scores_``; the ungrouped one on a tie). It costs one
                more search, and the lags and the horizon (given in rows) become that many cycles when they are whole numbers of them, else the series is not
                searched; a fit given a ``group_col``, ``architectures`` or ``difference`` is not searched. ``predict()`` adds the group column to the
                frame it is given. Dates that allow no grouping (irregular or coarser steps, more than 96 periods, too few days) leave the fit as it was.
            group_search_by (str): How ``group_search=True`` compares the two versions: ``"validation"`` (default) on the errors of the final models on
                the development and validation dates, or ``"rolling"`` on the rolling folds (needs ``rolling_folds``): the error of the best static
                expert of each version on the last horizons of the training rows, each predicted from everything before it. The validation error did
                not rank the versions on the hourly day-ahead load; the rolling folds average several horizons. ``group_choice_`` says which was used.
            rolling_folds (Optional[int]): Opt-in. The experts and the champions are scored on ``rolling_folds`` rolling origins instead of the usual
                folds: the last ``rolling_folds`` horizons of the training rows (the horizon is ``horizon``, else the smallest lag, else 1 date),
                each predicted by a model trained on everything before it. A selection on one horizon of data is noisy; the mean of several is
                steadier. Needs ``date_col``; with a ``'holdout'`` validation or too short a history the usual folds are used.
            dynamic_top (Optional[int]): Opt-in, to cut the cost of a fit. The Kalman and adaptive experts, which take most of the time, are built
                only for the champion formulas ranked at most this in their island (1: the best of each island); every champion still gets its
                static experts. ``None`` (default) builds them for all champions, the search as it was.
            horizon (Optional[int]): Number of steps ahead the model is asked to forecast. By default it is the smallest lag (the target is
                known up to that many steps before a forecast row). When given, every lag of the target must be at least the horizon (a
                shorter lag is not known that far ahead): ``fit()`` raises a ``ValueError`` otherwise. Without explicit lags, the lags of the
                dynamic experts become the horizon and, for steps of a day or less, one week.
            seed (Optional[int]): Makes the search reproducible. With an integer, ``fit()`` seeds the global generators of ``random``, ``numpy`` and
                ``torch``, runs, and restores them: the same seed on the same data gives the same experts, whatever was drawn before and the hash seed
                of the interpreter; the seed is written in the export next to the run. ``None`` (default) draws from the global generators as before.
                Not thread-safe. It does not make the numerical solver differ between CPU and GPU.
            replace_sentinels (bool): Opt-in. Values that repeat at an extreme of the training target, far outside its robust spread (a
                missing-value code such as -9999), are always reported (``target_profile_['sentinels']``, with a warning); with ``True`` they are
                replaced by missing values before the fit, otherwise they are fitted as values.
            seasonal_anchor (Optional[bool]): Opt-in anchor: an expert that regresses the target on its value one season earlier is added to the pool
                (island ``Anchor``, expanded like any champion), so the ensembles can always fall back on the last season repeated. ``None``
                (default) is off, except with ``search="fast"``, which turns it on; ``True`` and ``False`` are always obeyed.
            season (Optional[int]): Number of steps of one season for the anchor; by default the natural season of the sampling step (24 hourly,
                7 daily, 52 weekly, 12 monthly, 4 quarterly). It must be at least the horizon (the smallest lag of the target).
            extrapolation_guards (bool): Opt-in rules against extrapolation. Champions with more coefficients than half the fit rows, a neural or
                PID term on fewer than 1000 rows, or a linear term on a counter are refused; a never-negative target is floored at zero; the rows
                each expert had to be clipped on are counted against it in the selection and reported in ``clipped_rows_``.
            mandatory_variables (Optional[List[str]]): Variables required in all models.
            n_experts (int): Number of top expert architectures to retain in the ensemble.
            pop_size (int): Population size for island-based evolutionary exploration.
            use_opera (bool): Whether to use OPERA minimax aggregation for ensembling.
            eta (float): Learning rate for the estimation of distribution algorithm.
            complexity_penalty (float): Multiplier penalizing model structural complexity.
            export_dir (str): Directory where intermediate logs and exports are saved.
            **kwargs: Additional runtime options forwarded to sub-components.

        Raises:
            TypeError: If deprecated 'mandatory_terms' keyword argument is passed.
        """
        if "mandatory_terms" in kwargs:
            raise TypeError("AutoTAM.__init__() got an unexpected keyword argument 'mandatory_terms'")
        self._constructor_args = {**{k: v for k, v in locals().items() if k not in ("self", "kwargs", "__class__")}, **kwargs}
        self.search = check_search_mode(search)
        self.family = check_family_mode(family)
        self.family_search = bool(family_search)
        self.offset_col = offset_col
        self.strong_regularization = bool(strong_regularization)
        if autofit_criterion not in ("gcv", "reml"):
            raise ValueError(f"autofit_criterion must be 'gcv' or 'reml'; got {autofit_criterion!r}")
        self.autofit_criterion = autofit_criterion
        self.transform_experts = bool(transform_experts)
        self.family_adaptive = bool(family_adaptive)
        self.extended_effects = bool(extended_effects)
        if small_sample not in (True, False, "auto"):
            raise ValueError(f"small_sample must be False, True or 'auto'; got {small_sample!r}")
        self.small_sample = small_sample
        self.is_small_sample_ = False   # set by fit(): whether the small-sample protocol was used
        self.spatial_pairs_ = []          # set by fit() with spatial_pairs: the coordinate pairs found
        self.random_effect_columns_ = []  # set by fit() with random_effects: the identifier columns found
        if guided_search and self.family == "gaussian":
            self.family = "auto"
        fast = self.search == "fast"
        n_experts = n_experts if n_experts is not None else (FAST_N_EXPERTS if fast else 15)
        pop_size = pop_size if pop_size is not None else (FAST_POP_SIZE if fast else 64)
        self.anchor_from_preset = seasonal_anchor is None and fast   # then a step without a natural season only means no anchor
        seasonal_anchor = seasonal_anchor if seasonal_anchor is not None else fast
        if series_guard and not (pooled_floor or seasonal_anchor):
            raise ValueError("series_guard needs a guard expert: set pooled_floor=True (a seasonal profile) or seasonal_anchor=True (the last season).")
        self.formula = formula
        self.n_experts = n_experts
        self.complexity_penalty = complexity_penalty
        if isinstance(mandatory_variables, str):
            mandatory_variables = [mandatory_variables]
        self.mandatory_variables = mandatory_variables or []
        
        self.data_manager = DataManager(
            formula=formula,
            lags=lags,
            mandatory_variables=self.mandatory_variables,
            horizon=horizon,
            rolling_folds=rolling_folds,
            pooled_floor=pooled_floor,
            max_cv_folds=max_cv_folds if max_cv_folds is not None else (FAST_MAX_CV_FOLDS if fast else None),
            clean_sentinels=clean_sentinels,
            calendar_guards=calendar_guards,
            offset_col=offset_col,
            strong_regularization=strong_regularization,
            extended_effects=extended_effects,
            small_sample=small_sample is True,
            spatial_pairs=spatial_pairs,
            random_effects=random_effects,
            smooth_monotonic=smooth_monotonic,
            cross_sectional=cross_sectional,
            physical_bounds=physical_bounds,
            calendar_features=calendar_features,
            country=country,
            subdivision=subdivision,
            regimes=regimes,
            extended_search=extended_search,
            crisis_calendar=crisis_calendar,
        )
        self.mandatory_terms = self.data_manager.mandatory_terms
        self.discoverer = BaseDiscoverer(
            pop_size=pop_size, 
            eta=eta, 
            mandatory_terms=self.mandatory_terms,
            mandatory_variables=self.mandatory_variables,
            islands=FAST_ISLANDS if fast else None,
            n_generations=FAST_GENERATIONS if fast else FULL_GENERATIONS,
            extended=bool(extended_search),
            crossover=bool(crossover),
            bounded_rewards=bool(bounded_rewards),
            tensor_margin_penalty=bool(tensor_margin_penalty),
        )
        self.expander = ExpertExpander()
        self.selector = EnsembleSelector(use_opera=use_opera, apex_quality_ratio=apex_quality_ratio, weight_average=frozen_weights, seed_floor=seed_floor, pooled_floor=pooled_floor, apex_prune=apex_prune)
        self.reporter = EvolutionReporter(export_dir=export_dir)
        
        self.export_directory = export_dir
        self.ctx = None
        self.trained_experts = [] 
        self.league_weights = {}
        self.weights_top10 = {}
        self.island_aliases = {}
        self.oof_predictions_ = {}
        self.chronological_test_log = []
        self.online_weights_ = {}   # populated by predict_online(): MLpol weight trajectories
        self.apex_members_ = []     # the whole Apex pool, aggregated by predict_online()
        self.extrapolation_guards = bool(extrapolation_guards)
        self.seasonal_anchor = bool(seasonal_anchor)
        self.replace_sentinels = bool(replace_sentinels)
        self.seed = seed
        self.target_profile_ = {}  # set by fit(): sentinels, bounds, shape and a family hint of the training target
        self.season = season
        self.clipped_rows_ = {}  # with the guards: rows clipped, by expert, set by predict()
        self.final_model_column_ = None  # set by fit(): the column of predict() that holds the final model
        self.calendar_profile_ = {}  # set by fit() with calendar_features: the spectral profile of the training target
        self.search_groups = search_groups
        if search_sample not in ("level", "level_volatility"):
            raise ValueError(f"search_sample must be 'level' or 'level_volatility'; got {search_sample!r}")
        self.search_sample = search_sample
        if group_search_by not in ("validation", "rolling"):
            raise ValueError(f"group_search_by must be 'validation' or 'rolling'; got {group_search_by!r}")
        self.group_search_by = group_search_by
        self.group_choice_ = None        # set by fit() with group_search=True: how the versions were compared, 'validation' or 'rolling'
        if group_search not in (True, False, None, "always"):
            raise ValueError(f"group_search must be False, True or 'always'; got {group_search!r}")
        self.group_search = "always" if group_search == "always" else bool(group_search)
        self.online_variants = bool(online_variants)
        self.native_quantiles = bool(native_quantiles)
        self._native_levels = None       # the levels predict_quantiles asks the native experts for, during its predict() call
        self._native_quantiles_ = {}     # {expert: array rows x levels} of the last predict() with native levels
        self.quantile_members_ = []      # the members aggregated by the last predict_quantiles(aggregation="opera")
        self.derived_group_ = None       # set by fit() with group_search: {"column", "kind", "count"} of the grouping kept, else None
        self.group_scores_ = {}          # set by fit() with group_search: the validation error of each version
        self._choosing_group = False
        self.difference = difference if not (guided_search and difference is None) else "auto"
        if dynamic_top is not None and (isinstance(dynamic_top, bool) or int(dynamic_top) != dynamic_top or dynamic_top < 1):
            raise ValueError(f"dynamic_top must be None or a positive whole number, got {dynamic_top!r}.")
        self.dynamic_top = None if dynamic_top is None else int(dynamic_top)
        self.difference_ = None          # set by fit(): the lag of the difference in use, None for the level
        self.difference_scores_ = {}     # set by fit() with difference='select': the errors of the level and of the difference
        self._levels = None
        self._differenced_input = False
        self.guided_search = bool(guided_search)
        self.guided_hints_ = {}          # set by fit() with guided_search: the diagnostics that guided the fit and the advice after it
        self.diagnostics = bool(diagnostics) or self.guided_search
        self.target_diagnostics_ = {}    # set by fit() with diagnostics=True
        self.residual_profile_ = {}      # set by fit() with diagnostics=True
        self.search_sample_ = []  # set by fit() with search_groups: the groups the search ran on
        if max_covariates is not None and (isinstance(max_covariates, bool) or int(max_covariates) != max_covariates or max_covariates < 1):
            raise ValueError(f"max_covariates must be None or a positive whole number, got {max_covariates!r}.")
        self.rank_covariates = bool(rank_covariates) or max_covariates is not None
        self.cross_sectional = bool(cross_sectional)
        self.max_covariates = None if max_covariates is None else int(max_covariates)
        self.covariate_ranking_ = pd.DataFrame()  # set by fit() with rank_covariates: feature, score, redundant_with
        self.inactivity_gate = bool(inactivity_gate)
        self.closed_when = dict(closed_when or {})
        self.gate_ = None            # set by fit() with inactivity_gate: the learned InactivityGate
        self.inactive_groups_ = []   # set by fit() with inactivity_gate: groups whose last training values are zero
        self.series_guard = bool(series_guard)
        self.sentinel_report_ = {}  # set by fit() with clean_sentinels: per column, the codes, the range and the cells masked
        self.guarded_series_ = {}  # set by fit() with series_guard: {group: guard expert} for the groups the final model loses on
        self.apex_pruned_ = []  # set by fit() with apex_prune: the Apex members dropped for keeping no weight
        self.timings_ = {}     # set by fit(): seconds of the search, the expansion and the selection, and per series (group)
        self.experts_ = pd.DataFrame()  # set by fit(): one row per trained expert, with its stable identifier and its Apex weight
        self.architectures_ = {}  # set by fit(): the champion formulas by island, to pass as architectures= to a later fit
        self.history_ = None  # set by fit(): the training rows, put in front of a frame that starts after them
        self.extrapolating_features_ = []
#: </auto_tam_init>

#: <auto_tam_fit>
    @seeded_method
    def fit(
        self, 
        df_train: Optional[pd.DataFrame] = None, 
        df_fit: Optional[pd.DataFrame] = None, 
        df_dev: Optional[pd.DataFrame] = None, 
        df_val: Optional[pd.DataFrame] = None, 
        date_col: Optional[str] = None, 
        group_col: Optional[str] = None, 
        expansions: Optional[Dict[str, bool]] = None, 
        validation_strategy: str = 'auto',
        optimization_metric: str = 'rmse',
        refit_on_full_train: bool = False,
        past_covariates: Optional[Dict[str, List[int]]] = None,
        seed_formulas: Optional[List[str]] = None,
        architectures: Optional[Union['AutoTAM', Dict[str, List[str]]]] = None,
        external_experts: Optional[Dict[str, Any]] = None
    ):
        """Fits the search on the given data.

        Args:
            df_train: Training frame, split by fractions into fit, dev and val.
            df_fit: Fit rows, used with ``df_dev`` and ``df_val`` instead of ``df_train``.
            df_dev: Development rows.
            df_val: Validation rows.
            date_col: Name of the date column.
            group_col: Name of the group column (one model per group).
            expansions: Which dynamic and static expansions of the base models are tried.
            validation_strategy: ``'auto'``, ``'holdout'`` or ``'expanding_window'``.
            optimization_metric: Metric the experts and the ensembles are selected on: ``'rmse'`` (default), ``'mae'``, ``'mape'``, or the weighted
                quantile loss ``'wql'`` (levels 0.1, 0.5, 0.9) or ``'wql:0.05,0.5,0.95'``, the loss of the quantile layer of ``predict_quantiles``
                (the point forecast plus the quantiles of its own errors). The structure search itself keeps the point loss.
            refit_on_full_train: Refit the experts on the training rows before the final scoring.
            past_covariates: Past-only covariates, ``{column: [lag orders]}``: columns known up to the forecast origin only. The covariate itself is
                never a feature; ``<column>_lag_<order>`` is built per group like the lags of the target and offered to the search. Each order must
                be at least the horizon (the smallest lag of the target).
            seed_formulas: Formulas the user already trusts, ``["y ~ l(x1) + s(x2, k=10)"]``. Each is expanded like a champion of the search (prior,
                Kalman and adaptive experts) under the island name ``Seed``, so it competes in the leagues and in the Apex pool; the experts
                are named ``..._Seed_R<k>_...``. A formula with another target, an unknown column or a syntax error raises a ``ValueError``.
            architectures: A fitted ``AutoTAM`` or its ``architectures_`` (``{island: [formulas]}``). The evolutionary search is skipped: the
                champion formulas of that earlier search are expanded and refitted on the given rows (new coefficients, same experts), which is how
                one search serves many windows of a task. A formula that names a column absent from the new data is dropped with a warning; if
                none is left, a ``ValueError`` is raised. Use a new ``AutoTAM`` for the refit (same formula and settings as the first). It saves the
                search only: the expansion of the experts (prior, Kalman, adaptive) and their scoring, which take most of the time of a fit, still run,
                and a new search for each window was the more accurate in a measurement on weekly order counts and hourly prices.

            external_experts: Forecasts made outside AutoTAM, ``{name: forecast}``, that join the Apex pool and are aggregated with the experts
                by the same Opera (the caller's rules model, a baseline). A forecast is a ``pd.Series`` indexed by date (or by ``(date, group)``)
                or a ``pd.DataFrame`` with the date column, the group column when there is one, and ``forecast``; it must cover every validation
                row (the rows are looked up by date, so the split need not be known). They are never refitted. Their identifier is their name.
                Needs ``date_col`` and ``use_opera=True``; not available with ``difference`` or a derived group (``group_search``).

        Returns:
            The fitted model.

        Raises:
            ValueError: Invalid ``past_covariates``, an empty split, or no expert survives the search.
        """
        if external_experts:
            if self.difference is not None or (self.group_search and not group_col):
                raise NotImplementedError("external_experts is not available with difference=... or group_search (the rows they forecast change with the version).")
            if not self.selector.use_opera:
                raise ValueError("external_experts needs use_opera=True: the Apex aggregation is what weights them.")
        if self.group_search and not group_col and date_col and architectures is None and self.difference is None and not self._choosing_group:
            started_search = time.perf_counter()
            fitted = self._fit_choosing_group(df_train, df_fit, df_dev, df_val, date_col, expansions, validation_strategy, optimization_metric,
                                              refit_on_full_train, past_covariates, seed_formulas)
            fitted.timings_ = {**fitted.timings_, "group_search_total": time.perf_counter() - started_search}   # every version fitted, not only the kept one
            return fitted
        if self.search_groups and architectures is None and group_col:
            architectures = self._search_on_a_sample(df_train, df_fit, df_dev, df_val, date_col, group_col, expansions, validation_strategy,
                                                     optimization_metric, refit_on_full_train, past_covariates, seed_formulas)
        if self.difference is not None:
            fit_arguments = dict(expansions=expansions, validation_strategy=validation_strategy, optimization_metric=optimization_metric,
                                 refit_on_full_train=refit_on_full_train, past_covariates=past_covariates, seed_formulas=seed_formulas)
            df_train, df_fit, df_dev, df_val = self._difference_training_frames(df_train, df_fit, df_dev, df_val, date_col, group_col, fit_arguments)
        logger.info("AutoTAM: Starting Pipeline...")
        self.chronological_test_log = []
        if expansions is None and self.cross_sectional and not date_col:
            expansions = {"prior": True, "autofit": True, "kalman": False, "adaptive": False, "grid": False}
        if self.offset_col and expansions is None:
            expansions = {"prior": True, "autofit": True, "kalman": False, "adaptive": False, "grid": False}   # the engine refuses an offset in the online stages
        self.expander.expansions = expansions or {"prior": True, "autofit": True, "kalman": True, "adaptive": True, "grid": False}
        
        if self.inactivity_gate:
            target_name = self.data_manager.formula_config["targets"][0]
            df_train, df_fit, df_dev, df_val = (None if f is None else drop_leading_zeros(f, target_name, group_col, date_col)
                                                for f in (df_train, df_fit, df_dev, df_val))
        self.ctx = self.data_manager.prepare(
            df_train=df_train, 
            df_fit=df_fit, 
            df_dev=df_dev, 
            df_val=df_val, 
            date_col=date_col, 
            group_col=group_col,
            validation_strategy=validation_strategy,
            past_covariates=past_covariates,
            seasonal_anchor="auto" if (self.seasonal_anchor and self.anchor_from_preset) else self.seasonal_anchor,
            season=self.season,
            replace_sentinels=self.replace_sentinels
        )
        self.target_profile_ = self.ctx.target_profile
        if self.diagnostics:
            self.target_diagnostics_ = self._target_diagnostics()
        self.history_ = self._training_rows(df_train, df_fit, df_dev, df_val)
        seed_champions = self._seed_champions(seed_formulas)
        if self.data_manager.calendar_features:
            seed_champions = list(seed_champions) + self._calendar_seeds()
        if self.ctx.spatial_pairs:
            seed_champions = list(seed_champions) + self._spatial_seeds()
        self.ctx.extrapolation_guards = self.extrapolation_guards
        self.ctx.fast_search = self.search == "fast"
        self.ctx.dynamic_top = self.dynamic_top
        if self.rank_covariates:
            self._rank_covariates()
        self.ctx.cycle_windows = group_col == GROUP_COLUMN   # a derived group steps by cycle: the adaptive windows are counted in cycles of one group
        self.ctx.online_variants = self.online_variants
        self.ctx.native_quantiles = self.native_quantiles
        self.ctx.family_losses = candidate_losses(self.ctx.target_profile) if self.family == "auto" else []
        self.ctx.family_search = self.family_search
        self.ctx.strong_regularization = self.strong_regularization
        self.ctx.autofit_criterion = self.autofit_criterion
        self.ctx.transform_experts = self.transform_experts
        self.ctx.family_adaptive = self.family_adaptive
        self.ctx.extended_effects = self.extended_effects
        rows = sum(len(f) for f in (self.ctx.df_fit, self.ctx.df_dev, self.ctx.df_val) if f is not None)
        self.is_small_sample_ = self.small_sample is True or (self.small_sample == "auto" and not self.ctx.date_col and rows < 150)
        if self.is_small_sample_:
            self.ctx.small_sample_rows = rows
            self.ctx.search_space["max_terms"] = max(1, rows // 8)
            self.selector.seed_floor = True
            seed_champions = list(seed_champions) + self._small_sample_baseline()
        self.spatial_pairs_, self.random_effect_columns_ = list(self.ctx.spatial_pairs), list(self.ctx.random_effect_columns)
        if self.guided_search:
            self.guided_hints_ = self._guided_hints()
        self.ctx.optimization_metric = optimization_metric
        self.ctx.complexity_penalty = self.complexity_penalty
        
        # The search fits and scores hundreds of models: their extrapolation warnings are collected, not raised one by one.
        kept_architectures = self._kept_architectures(architectures)
        started = time.perf_counter()
        with collect_extrapolation() as extrapolating:
            if kept_architectures is None:
                island_champions, draga_engine = self.discoverer.search(self.ctx)
            else:
                island_champions, draga_engine = kept_architectures, None
            if seed_champions:
                island_champions = {**island_champions, "Seed": seed_champions}
            if self.ctx.anchor_season:
                island_champions = {**island_champions, ANCHOR_ISLAND: [anchor_formula(self.ctx.target, self.ctx.anchor_season)]}
            if self.ctx.floor:
                island_champions = {**island_champions, FLOOR_ISLAND: [floor_formula(self.ctx.target, self.ctx.floor["count"])]}
            searched = time.perf_counter()
            self.architectures_ = {island: list(formulas) for island, formulas in island_champions.items()}
            if draga_engine is not None:
                self.reporter.export_evolutionary_diagnostics(draga_engine)

            candidate_pool = self.expander.generate_experts(island_champions, self.ctx, self.chronological_test_log, self.reporter)

            expanded = time.perf_counter()
            if not candidate_pool:
                raise ValueError(self._no_expert_message())

            self.trained_experts, self.league_weights, self.weights_top10, self.island_aliases, self.oof_predictions_ = self.selector.evaluate_and_refit(
                candidate_pool, self.ctx, self.chronological_test_log, self.reporter, self.expander, refit_on_full_train=refit_on_full_train,
                external=normalise_external(external_experts, self.ctx.date_col, self.ctx.group_col, taken=list(candidate_pool))
            )
        selected = time.perf_counter()
        self.apex_members_ = list(getattr(self.selector, "apex_members_", []))
        self.timings_ = self._timings(started, searched, expanded, selected)
        self.apex_pruned_ = list(self.selector.apex_pruned_)
        self.final_model_column_ = self.FINAL_MODEL_COLUMN if self.weights_top10 else None
        self.experts_ = experts_table(self.trained_experts, self.weights_top10)
        if self.inactivity_gate:
            training = pd.concat([f for f in (self.ctx.df_fit, self.ctx.df_dev, self.ctx.df_val) if f is not None])
            self.gate_ = InactivityGate(self.ctx.target, self.ctx.date_col, self.ctx.group_col, self.closed_when).fit(training)
            self.inactive_groups_ = list(self.gate_.inactive_at_cutoff)
        cleaner = getattr(self.ctx, "sentinel_cleaner", None)
        self.sentinel_report_ = cleaner.report(self.history_) if cleaner is not None and self.history_ is not None else {}
        self.guarded_series_ = self._guard_choice() if self.series_guard else {}
        if self.diagnostics:
            self.residual_profile_ = self._residual_profile()
        if self.guided_search:
            self.guided_hints_["advice"] = residual_advice(self.residual_profile_)
            for sentence in self.guided_hints_["advice"]:
                logger.info("AutoTAM guided search: %s", sentence)
        self.extrapolating_features_ = sorted(extrapolating)
        
        if not self.trained_experts:
            raise ValueError(self._no_expert_message())

        self.reporter.export_final_architectures(self.trained_experts)
        if self.seed is not None:
            extra = {"experts": self.experts_.to_dict(orient="records"), "timings": self.timings_}
            if self.series_guard:
                extra["guarded_series"] = {str(k): v for k, v in self.guarded_series_.items()}
            if self.diagnostics:
                extra["diagnostics"] = {"target": self.target_diagnostics_, "residuals": self.residual_profile_}
            self.reporter.export_run_info(self.seed, extra)
        self.reporter.export_chronological_tests(self.chronological_test_log)
        self.reporter.export_collinearity_purge_log(self.data_manager.engineer.purge_log)
        return self
#: </auto_tam_fit>

#: <auto_tam_diagnostics>
    def _timings(self, started: float, searched: float, expanded: float, selected: float) -> Dict[str, float]:
        """Seconds of the phases of the last ``fit()``, and the cost per series (the groups, or 1)."""
        group_col = self.ctx.group_col
        frames = [f for f in (self.ctx.df_fit, self.ctx.df_dev, self.ctx.df_val) if f is not None]
        series = max(1, int(pd.concat(frames)[group_col].nunique())) if group_col and frames and group_col in frames[0].columns else 1
        total = selected - started
        return {"search": searched - started, "expansion": expanded - searched, "selection": selected - expanded, "total": total,
                "series": series, "seconds_per_series": total / series}

    def _guard_choice(self) -> Dict[Any, str]:
        """The groups whose validation RMSE is smaller for a guard expert (``Floor`` and ``Anchor`` islands) than for the final model."""
        names = [e["name"] for e in self.trained_experts if e.get("island") in GUARD_ISLANDS and e["name"] in self.oof_predictions_]
        column = self.final_model_column_
        if not names:
            raise ValueError("series_guard needs a guard expert: set pooled_floor=True (a seasonal profile) or seasonal_anchor=True (the last season).")
        if column is None or column not in self.oof_predictions_:
            return {}
        val = self.ctx.df_val
        tail = lambda name: np.asarray(self.oof_predictions_[name], dtype=float)[-len(val):]
        group_col = self.ctx.group_col
        groups = val[group_col].to_numpy() if group_col and group_col in val.columns else None
        chosen = guarded_series(val[self.ctx.target].to_numpy(dtype=float), tail(column), {n: tail(n) for n in names}, groups)
        if chosen:
            logger.info("AutoTAM series guard: %d group(s) are forecast by a guard expert (it beat the final model on their validation rows).", len(chosen))
        return chosen

    def _guard_columns(self, preds_df: pd.DataFrame, df_test_clean: pd.DataFrame) -> pd.DataFrame:
        """Replaces the final model's columns by the guard expert's on the rows of the guarded groups."""
        if not self.guarded_series_:
            return preds_df
        group_col = self.ctx.group_col
        groups = df_test_clean[group_col].to_numpy() if group_col and group_col in df_test_clean.columns else None
        if groups is not None and len(groups) != len(preds_df):
            return preds_df
        for column in (self.FINAL_MODEL_COLUMN, self.FINAL_MODEL_COLUMN_ONLINE):
            preds_df = apply_guard(preds_df, column, self.guarded_series_, groups)
        return preds_df

    def _rank_covariates(self) -> None:
        """Scores the covariates of the search space, drops the extra ones when ``max_covariates`` is set, gives the rest an importance."""
        ctx = self.ctx
        frames = [f for f in (ctx.df_fit, ctx.df_dev, ctx.df_val) if f is not None]
        rows = pd.concat(frames)
        calendar = set(ctx.calendar.get("columns", [])) if ctx.calendar else set()
        derived = (f"{ctx.target}_lag_", "anchor_lag_", "crisis_", FLOOR_COLUMN, "regime")
        candidates = [f for f in ctx.search_space if f in rows.columns and f not in (ctx.target, ctx.date_col, ctx.group_col)
                      and f not in calendar and not str(f).startswith(derived)]
        table = rank_covariates(rows, ctx.target, candidates, ctx.group_col)
        mandatory = set(ctx.mandatory_variables or [])
        keep = [f for f, twin in zip(table["feature"], table["redundant_with"]) if twin is None or f in mandatory]
        if self.max_covariates is not None:
            chosen = [f for f in keep if f not in mandatory][: max(0, self.max_covariates - len(mandatory))] + [f for f in keep if f in mandatory]
            for feature in [f for f in candidates if f not in chosen and f not in mandatory]:
                ctx.search_space.pop(feature, None)
            keep = chosen
        for feature, score in zip(table["feature"], table["score"]):
            if feature in ctx.search_space and feature in keep:
                ctx.search_space[feature]["importance"] = float(score)
        table["offered"] = table["feature"].isin([f for f in table["feature"] if f in ctx.search_space])
        self.covariate_ranking_ = table
        logger.info("AutoTAM covariate ranking: %s", ", ".join(f"{f} {s:.2f}" for f, s in zip(table["feature"], table["score"])))

    def _spatial_seeds(self) -> List[str]:
        """Seed formulas for the coordinate pairs found: a thin-plate surface and a tensor product of two smooths, with the exposure offset."""
        seeds = []
        for first, second in self.ctx.spatial_pairs:
            for surface in (f"rbf({first}, others='{second}', n_centers=30, kernel='tps', ap=-2.0)",
                            f"te(s({first}, k=8, ap=-2.0), s({second}, k=8, ap=-2.0), ap=0.0)"):
                seeds.append(with_offset(f"{self.ctx.target} ~ {surface}", self.ctx.offset_col))
        return seeds

    def _small_sample_baseline(self) -> List[str]:
        """The floor of a small sample: a heavily regularised line in each of the first three numeric features (a seed expert the final model
        is never worse than on the validation rows)."""
        ctx = self.ctx
        names = [c for c in ctx.formula_config.get("features", []) if c in ctx.df_fit.columns and c not in (ctx.target, ctx.date_col, ctx.group_col)
                 and pd.api.types.is_numeric_dtype(ctx.df_fit[c])][:3]
        return [f"{ctx.target} ~ " + " + ".join(f"l({c}, ap=0.0)" for c in names)] if names else []

    def _gated(self, values: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
        """``values`` with the rows of the inactive series set to zero (one row of ``values`` per row of ``frame``)."""
        return self.gate_.apply(values, frame) if self.gate_ is not None else values

    def _guided_hints(self) -> Dict[str, Any]:
        """What the diagnostics of the training rows say before the search: the verdict, the hints applied, the weak features."""
        target = self.ctx.target
        stationarity = self.target_diagnostics_.get("stationarity")
        if isinstance(stationarity, dict):
            stationarity = stationarity.get("verdict")
        horizon = int(self.ctx.requested_horizon or min(self.data_manager.user_lags or [1]))
        features = [f for f in self.ctx.formula_config.get("features", []) if f in self.ctx.df_fit.columns]
        screened = feature_diagnostics(self.ctx.df_fit, target, features, horizon, self.ctx.group_col)
        weak = sorted(name for name, row in screened.items() if row["weak"])
        hints = {"stationarity": stationarity, "difference": self.difference, "family_losses": list(self.ctx.family_losses),
                 "features": screened, "weak_features": weak, "advice": []}
        logger.info("AutoTAM guided search: stationarity %s, difference %s, families %s, %d weak feature(s)%s.", stationarity, self.difference,
                    hints["family_losses"] or "none", len(weak), f" ({', '.join(weak)})" if weak else "")
        return hints

    def _target_diagnostics(self) -> Dict[str, Any]:
        """Diagnostics of the training target: of the series itself, or of up to ten groups (the verdict is then the most frequent one)."""
        frames = [f for f in (self.ctx.df_fit, self.ctx.df_dev, self.ctx.df_val) if f is not None]
        rows = pd.concat(frames)
        rows = rows[rows[self.ctx.target].notna()]
        period = natural_season(self.ctx.metadata.get("delta_t", "1D"))
        group_col = self.ctx.group_col
        if not (group_col and group_col in rows.columns):
            return target_diagnostics(rows[self.ctx.target].to_numpy(), period)
        keys = sorted(rows[group_col].unique(), key=str)
        keys = [keys[int(i)] for i in np.linspace(0, len(keys) - 1, min(10, len(keys))).round()]
        by_group = {str(k): target_diagnostics(rows.loc[rows[group_col] == k, self.ctx.target].to_numpy(), period) for k in keys}
        verdicts = [d["stationarity"]["verdict"] for d in by_group.values() if "stationarity" in d]
        return {"groups": by_group, "stationarity": max(set(verdicts), key=verdicts.count) if verdicts else None}

    def _residual_profile(self) -> Dict[str, Any]:
        """Diagnostics of the errors of the final model on the validation rows (empty without a final model)."""
        column = self.final_model_column_
        if column is None or column not in self.oof_predictions_:
            return {}
        val = self.ctx.df_val
        in_val = np.asarray(self.oof_predictions_[column], dtype=float)[-len(val):]
        errors = val[self.ctx.target].to_numpy(dtype=float) - in_val
        return residual_diagnostics(errors, horizon=getattr(self.ctx, "inferred_horizon", 1))
#: </auto_tam_diagnostics>

#: <auto_tam_difference>
    def _difference_lag(self, frames, target: str, date_col: Optional[str] = None, group_col: Optional[str] = None) -> Optional[int]:
        """The lag of the difference to use: the number given, or for ``"auto"`` the horizon when the training target is cumulative, monotone
        or has a unit root (the verdict of the Dickey-Fuller and KPSS tests, in most of up to five groups)."""
        if self.difference == "auto":
            training = pd.concat([f for f in frames if f is not None])
            if profile_target(training[target])["hints"][0] != "cumulative" and not self._has_unit_root(training, target, date_col, group_col):
                return None
            return int(self.data_manager.horizon or min(self.data_manager.user_lags or [1]))
        if isinstance(self.difference, bool) or not isinstance(self.difference, (int, np.integer)):
            raise ValueError(f"difference must be None, \"auto\", \"select\" or a whole number of steps, got {self.difference!r}.")
        return int(self.difference)

    def _select_difference(self, frames, target: str, date_col: Optional[str], group_col: Optional[str], fit_arguments: Dict[str, Any]) -> Optional[int]:
        """Level or difference, chosen on forecasts the models did not see: the lag of the difference, or ``None`` for the level.

        The training rows are cut at their last three horizons. A level model and a differenced model are fitted on the rows before, each
        forecasts the three horizons in turn (the rows before each one observed, the horizon itself hidden), and the one with the smaller mean
        absolute error, in the level scale, wins. The scores are kept in ``difference_scores_``.
        """
        if not date_col:
            raise ValueError("difference needs a date column: pass date_col=.")
        h = int(self.data_manager.horizon or min(self.data_manager.user_lags or [1]))
        training = pd.concat([f for f in frames if f is not None])
        dates = np.sort(pd.to_datetime(training[date_col]).unique())
        blocks = 3
        if len(dates) < 4 * blocks * h + 10:
            logger.info("Difference not selected: %d dates are too few to hold out three horizons of %d.", len(dates), h)
            return None
        stamps = pd.to_datetime(training[date_col])
        edges = [dates[len(dates) - (blocks - i) * h] for i in range(blocks)] + [None]
        early = training[stamps < edges[0]]
        scores: Dict[str, float] = {}
        for label, mode in (("level", None), ("difference", h)):
            arguments = {**self._constructor_args, "difference": mode, "search_groups": None, "guided_search": False,
                         "export_dir": str(pathlib.Path(self.export_directory) / f"select_{label}")}
            child = type(self)(**arguments)
            child.fit(early, date_col=date_col, group_col=group_col, **fit_arguments)
            column = child.final_model_column_ or child.trained_experts[0]["name"]
            errors = []
            for i in range(blocks):
                upper = dates[len(dates) - (blocks - i - 1) * h - 1]
                window = training[(stamps >= edges[0]) & (stamps <= upper)].copy()
                hidden = pd.to_datetime(window[date_col]) >= edges[i]
                truth = window.loc[hidden, target].to_numpy(dtype=float, copy=True)   # a copy: the target is hidden just below
                window.loc[hidden, target] = np.nan
                forecast = child.predict(window, date_col=date_col)[column].to_numpy(dtype=float)[hidden.to_numpy()]
                errors.append(float(np.nanmean(np.abs(forecast - truth))))
            scores[label] = float(np.mean(errors))
        self.difference_scores_ = scores
        logger.info("Difference selection (mean absolute error of %d forecasts of %d steps): level %.5g, difference %.5g.", blocks, h, scores["level"], scores["difference"])
        return h if scores["difference"] < scores["level"] else None

    @staticmethod
    def _has_unit_root(training: pd.DataFrame, target: str, date_col: Optional[str], group_col: Optional[str]) -> bool:
        """Whether the unit-root verdict holds for most of up to five series (the groups, or the series itself)."""
        if group_col and group_col in training.columns:
            keys = sorted(training[group_col].unique(), key=str)
            keys = [keys[int(i)] for i in np.linspace(0, len(keys) - 1, min(5, len(keys))).round()]
            series = [training.loc[training[group_col] == k] for k in keys]
        else:
            series = [training]
        verdicts = []
        for part in series:
            part = part.sort_values(date_col) if date_col and date_col in part.columns else part
            values = part[target].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if len(values) >= 30:
                verdicts.append(stationarity_verdict(values)["verdict"] == "unit_root")
        return bool(verdicts) and sum(verdicts) * 2 > len(verdicts)

    def _difference_training_frames(self, df_train, df_fit, df_dev, df_val, date_col, group_col, fit_arguments=None):
        """Replaces the target of the training frames by its increments (see ``differencing.py``) and keeps the levels apart."""
        target = self.data_manager.parser.parse(self.formula)["targets"][0]
        self.difference_, self._levels = None, None
        if self.difference == "select":
            h = self._select_difference([df_train, df_fit, df_dev, df_val], target, date_col, group_col, fit_arguments or {})
        else:
            h = self._difference_lag([df_train, df_fit, df_dev, df_val], target, date_col, group_col)
        if h is None:
            logger.info("Difference not used: the training target is not cumulative or monotone.")
            return df_train, df_fit, df_dev, df_val
        frames, levels = difference_training({"train": df_train, "fit": df_fit, "dev": df_dev, "val": df_val}, target, date_col, group_col, h)
        self.difference_, self._levels, self._difference_target = h, levels, target
        logger.info("AutoTAM: the model learns the increments over %d step(s).", h)
        return frames["train"], frames["fit"], frames["dev"], frames["val"]
#: </auto_tam_difference>

#: <auto_tam_group_search>
    def _fit_choosing_group(self, df_train, df_fit, df_dev, df_val, date_col, expansions, validation_strategy, optimization_metric,
                            refit_on_full_train, past_covariates, seed_formulas):
        """Fits with the group derived from the dates and, unless ``group_search="always"``, without it, and keeps the better version."""
        arguments = dict(date_col=date_col, expansions=expansions, validation_strategy=validation_strategy, optimization_metric=optimization_metric,
                         refit_on_full_train=refit_on_full_train, past_covariates=past_covariates, seed_formulas=seed_formulas)
        frames = [df_train, df_fit, df_dev, df_val]
        always = self.group_search == "always"
        self.derived_group_, self.group_scores_ = None, {}

        def fit_ungrouped():
            self._choosing_group = True
            try:
                self.fit(*frames, group_col=None, **arguments)
            finally:
                self._choosing_group = False
            return self

        known = [f for f in frames if f is not None and date_col in f.columns]
        choice = derived_group(pd.concat(known)[date_col]) if known else None
        if choice is None:
            logger.info("AutoTAM group search: the dates allow no derived group (irregular or coarser step, too many periods or too few days).")
            return fit_ungrouped()
        kind, count = choice
        arguments_child = {**self._constructor_args, "group_search": False, "export_dir": str(pathlib.Path(self.export_directory) / "group_search")}
        # One step of a group is one cycle (a day): a lag or a horizon given in rows becomes that many cycles, when it is a whole number of them.
        lags, horizon = self._constructor_args.get("lags"), self._constructor_args.get("horizon")
        if lags:
            cycles = sorted({int(lag) // count for lag in lags if int(lag) % count == 0 and int(lag) >= count})
            if not cycles:
                logger.warning("AutoTAM group search: no lag in %s is a whole number of cycles of %d rows; the ungrouped fit is kept.", list(lags), count)
                return fit_ungrouped()
            arguments_child["lags"] = cycles
        if horizon:
            if int(horizon) % count or int(horizon) < count:
                logger.warning("AutoTAM group search: the horizon of %d rows is not a whole number of cycles of %d rows; the ungrouped fit is kept.", horizon, count)
                return fit_ungrouped()
            arguments_child["horizon"] = int(horizon) // count
        child_arguments = dict(arguments)
        season = self._constructor_args.get("season")
        if season:
            cycles = to_cycles(season, count)
            if cycles is None:
                logger.warning("AutoTAM group search: the season of %d rows is not a whole number of cycles of %d rows; the ungrouped fit is kept.", season, count)
                return fit_ungrouped()
            arguments_child["season"] = cycles
        if seed_formulas:
            carried = [(formula, formula_in_cycles(formula, count)) for formula in seed_formulas]
            for formula, converted in carried:
                if converted is None:
                    logger.warning("AutoTAM group search: the seed formula %r has a lag that is not a whole number of cycles of %d rows; it is left out of the grouped fit.", formula, count)
            child_arguments["seed_formulas"] = [converted for _, converted in carried if converted is not None]
            if not child_arguments["seed_formulas"]:
                logger.warning("AutoTAM group search: no seed formula can be written in cycles; the ungrouped fit is kept.")
                return fit_ungrouped()
        if past_covariates:
            child_arguments["past_covariates"] = covariates_in_cycles(past_covariates, count)
            if not child_arguments["past_covariates"]:
                logger.warning("AutoTAM group search: no past-covariate order is a whole number of cycles of %d rows; the ungrouped fit is kept.", count)
                return fit_ungrouped()
        grouped = [None if f is None else add_group(f, date_col, kind, count) for f in frames]
        child = type(self)(**arguments_child)
        try:
            child.fit(*grouped, group_col=GROUP_COLUMN, **child_arguments)
        except Exception as exc:
            log_failure(logger, f"the fit with the derived group ({kind}, {count} groups)", exc)
            return fit_ungrouped()

        def adopt():
            kept = {k: getattr(self, k) for k in ("_constructor_args", "group_search", "group_search_by", "export_directory", "reporter", "group_scores_", "group_choice_", "_choosing_group")}
            self.__dict__.update(child.__dict__)
            self.__dict__.update(kept)
            self.derived_group_ = {"column": GROUP_COLUMN, "kind": kind, "count": count}

        if always:
            logger.info("AutoTAM group search: the series is fitted as %d groups (%s), as asked (group_search='always').", count, kind)
            adopt()
            return self
        fit_ungrouped()
        self.group_choice_ = "validation"
        if self.group_search_by == "rolling":
            scores = self._rolling_scores(self, child, kind)
            if scores is not None:
                self.group_scores_, self.group_choice_ = scores, "rolling"
                logger.info("AutoTAM group search (best static expert on the same rolling folds): ungrouped %.5g, %s (%d groups) %.5g.",
                            scores["ungrouped"], kind, count, scores[kind])
                if scores[kind] < scores["ungrouped"]:
                    adopt()
                return self
        ungrouped, derived = self._validation_errors_by_date(self), self._validation_errors_by_date(child)
        common = ungrouped.index.intersection(derived.index)
        if len(common) < 5:
            logger.warning("AutoTAM group search: too few common validation dates (%d) to compare; the ungrouped fit is kept.", len(common))
            return self
        scores = {"ungrouped": float(np.sqrt(np.mean(ungrouped.loc[common] ** 2))), kind: float(np.sqrt(np.mean(derived.loc[common] ** 2)))}
        self.group_scores_ = scores
        logger.info("AutoTAM group search (RMSE on %d common development and validation dates): ungrouped %.5g, %s (%d groups) %.5g.",
                    len(common), scores["ungrouped"], kind, count, scores[kind])
        if scores[kind] < scores["ungrouped"]:
            adopt()
        return self

    @staticmethod
    def _rolling_scores(ungrouped: "AutoTAM", grouped: "AutoTAM", kind: str) -> Optional[Dict[str, float]]:
        """The two versions compared on their rolling folds: the error of the best static expert of each, on folds that hold the same dates.

        The folds are cut from the last dates of the training rows, so the two versions are scored on the same period when both have rolling
        folds of the same size; ``None`` (with a warning) when one of them has none, or they differ: the comparison on the validation rows is used.
        """
        versions = {"ungrouped": ungrouped, kind: grouped}
        dates, scores = {}, {}
        for label, model in versions.items():
            folds = model.ctx.cv_folds or []
            date_col = model.ctx.date_col
            if not folds or not model.data_manager.rolling_folds or not date_col:
                logger.warning("AutoTAM group search: group_search_by='rolling' needs rolling_folds and a date column; the validation rows decide.")
                return None
            dates[label] = frozenset(pd.to_datetime(pd.concat([f[1] for f in folds])[date_col]).dt.floor("D").unique())   # a derived group steps by day
            static = [e["cv_rmse"] for e in model.trained_experts if e["type"] == "static" and pd.notna(e.get("cv_rmse"))]
            if not static:
                return None
            scores[label] = float(min(static))
        if dates["ungrouped"] != dates[kind]:
            logger.warning("AutoTAM group search: the rolling folds of the two versions do not hold the same dates; the validation rows decide.")
            return None
        return scores

    @staticmethod
    def _validation_errors_by_date(model: "AutoTAM") -> pd.Series:
        """The errors of the final model on its development and validation rows, indexed by date (the first expert without a final model).

        Both blocks are rows the static models were not fitted on; together they are twice the rows of the validation block alone, and a
        three-month block was not a reliable ground for the choice (the grouped fit was dropped on a seed where it won the hold-out by 10 %).
        """
        column = model.final_model_column_
        if column is None or column not in model.oof_predictions_:
            column = next(iter(model.oof_predictions_), None)
        blocks = [f for f in (model.ctx.df_dev, model.ctx.df_val) if f is not None and len(f)]
        if column is None or not blocks:
            return pd.Series(dtype=float)
        rows = pd.concat(blocks)
        predicted = np.asarray(model.oof_predictions_[column], dtype=float)[-len(rows):]
        errors = pd.Series(rows[model.ctx.target].to_numpy(dtype=float) - predicted, index=pd.to_datetime(rows[model.ctx.date_col]).to_numpy())
        return errors[np.isfinite(errors.to_numpy())].groupby(level=0).mean()

    def _with_derived_group(self, frame: pd.DataFrame) -> pd.DataFrame:
        """``frame`` with the derived group column when the model keeps one and the frame lacks it."""
        choice = self.derived_group_
        date_col = self.ctx.date_col if self.ctx is not None else None
        if not choice or choice["column"] in frame.columns or not date_col or date_col not in frame.columns:
            return frame
        return add_group(frame, date_col, choice["kind"], choice["count"])
#: </auto_tam_group_search>

#: <auto_tam_sample_search>
    def _search_on_a_sample(self, df_train, df_fit, df_dev, df_val, date_col, group_col, expansions, validation_strategy, optimization_metric,
                            refit_on_full_train, past_covariates, seed_formulas) -> Optional["AutoTAM"]:
        """Searches on a sample of the groups and returns the fitted searcher (``None`` when there are not more groups than the sample).

        The sample takes groups evenly spaced by the mean of their target, so small and large series are both in it; the choice is the same on
        every machine. The searcher is a new ``AutoTAM`` with the same arguments (it exports to ``<export_dir>/sample_search``).
        """
        frames = [f for f in (df_train, df_fit, df_dev, df_val) if f is not None and group_col in f.columns]
        everything = pd.concat(frames)
        target = self.data_manager.parser.parse(self.formula)["targets"][0]
        n_groups = everything[group_col].nunique()
        if n_groups <= self.search_groups:
            return None
        self.search_sample_ = sample_groups(everything, group_col, target, self.search_groups, self.search_sample)
        logger.info("AutoTAM: searching on %d of %d groups, then fitting all of them.", len(self.search_sample_), n_groups)

        def only(frame):
            return None if frame is None else frame[frame[group_col].isin(self.search_sample_)]

        args = {**self._constructor_args, "search_groups": None, "export_dir": str(pathlib.Path(self.export_directory) / "sample_search")}
        searcher = type(self)(**args)
        searcher.fit(only(df_train), only(df_fit), only(df_dev), only(df_val), date_col=date_col, group_col=group_col, expansions=expansions,
                     validation_strategy=validation_strategy, optimization_metric=optimization_metric, refit_on_full_train=refit_on_full_train,
                     past_covariates=past_covariates, seed_formulas=seed_formulas)
        return searcher
#: </auto_tam_sample_search>

#: <auto_tam_architectures>
    def _kept_architectures(self, architectures) -> Optional[Dict[str, List[str]]]:
        """The champion formulas of an earlier search, checked against the columns of the data given to this ``fit()``.

        Args:
            architectures: ``None`` (search as usual), a fitted ``AutoTAM`` or its ``architectures_`` dictionary.

        Returns:
            ``None`` when no architectures are given, else ``{island: [formulas]}`` without the formulas that name a missing column.

        Raises:
            ValueError: ``architectures`` is not a fitted ``AutoTAM`` or a dictionary of formula lists, or no formula can be used on the data.
        """
        if architectures is None:
            return None
        if isinstance(architectures, AutoTAM):
            architectures = getattr(architectures, "architectures_", None)
            if not architectures:
                raise ValueError("architectures: the AutoTAM given is not fitted (it has no architectures_).")
        if not isinstance(architectures, dict) or not all(isinstance(f, (list, tuple)) for f in architectures.values()):
            raise ValueError("architectures must be a fitted AutoTAM or a dictionary {island: [formulas]}.")
        columns = set(self.ctx.df_all_aug.columns)
        kept: Dict[str, List[str]] = {}
        dropped = 0
        for island, formulas in architectures.items():
            for formula in formulas:
                _, terms = parse_formula_to_terms(formula)
                names = set(term_columns(terms))
                if any(n and n not in columns for n in names):
                    dropped += 1
                else:
                    kept.setdefault(island, []).append(formula)
        if dropped:
            logger.warning("architectures: %d formula(s) of the earlier search name a column that is not in the data and were dropped.", dropped)
        if not kept:
            raise ValueError("architectures: no formula of the earlier search can be used on these data (their columns are missing).")
        logger.info("AutoTAM: refitting %d formulas of an earlier search, no evolutionary search.", sum(len(v) for v in kept.values()))
        return kept
#: </auto_tam_architectures>

#: <auto_tam_calendar_seeds>
    def _calendar_seeds(self) -> List[str]:
        """Profiles the cycles of the training target and returns the seed formulas for the ones present (empty when none)."""
        rows = pd.concat([self.ctx.df_fit, self.ctx.df_dev, self.ctx.df_val])   # every training row, never a forecast row
        rows = rows[rows[self.ctx.target].notna()]
        self.calendar_profile_ = spectral_profile(rows, self.ctx.target, self.ctx.date_col, self.ctx.group_col)
        columns = list(self.ctx.calendar.get("columns", []))
        seeds = calendar_seed_formulas(self.ctx.target, self.calendar_profile_, columns, holidays_on=bool(self.ctx.calendar.get("country")))
        logger.info("Calendar profile: cycles %s, %d seed formula(s).", sorted(self.calendar_profile_.get("cycles", {})), len(seeds))
        return seeds
#: </auto_tam_calendar_seeds>

#: <auto_tam_seed_formulas>
    def _seed_champions(self, seed_formulas: Optional[List[str]]) -> List[str]:
        """Validates the seed formulas given to ``fit()`` before the search and returns them as a list.

        Args:
            seed_formulas: Formulas the user already trusts, for example ``["y ~ l(x1) + s(x2, k=10)"]``.

        Returns:
            The formulas, in the order given (empty when ``seed_formulas`` is ``None``).

        Raises:
            ValueError: ``seed_formulas`` is not a list of strings, a formula cannot be parsed, its target is not the one of the AutoTAM
                formula, or it names a column that is not in the data.
        """
        if seed_formulas is None:
            return []
        if not isinstance(seed_formulas, (list, tuple)):
            raise ValueError(f"seed_formulas must be a list of formula strings, got {type(seed_formulas).__name__}.")
        columns = set(self.ctx.df_all_aug.columns)
        for formula in seed_formulas:
            if not isinstance(formula, str):
                raise ValueError(f"Each seed formula must be a string, got {formula!r}.")
            try:
                target, terms = parse_formula_to_terms(formula)
            except ValueError as exc:
                raise ValueError(f"Seed formula {formula!r} is not valid: {exc}") from exc
            if target != self.ctx.target:
                raise ValueError(f"Seed formula {formula!r} has target '{target}', the target of the AutoTAM formula is '{self.ctx.target}'.")
            names = set(term_columns(terms))
            missing = sorted(n for n in names if n and n not in columns)
            if missing:
                raise ValueError(f"Seed formula {formula!r} uses columns that are not in the data: {missing}.")
        return list(seed_formulas)
#: </auto_tam_seed_formulas>

#: <auto_tam_no_expert>
    def _no_expert_message(self) -> str:
        """Message of the error raised when the search keeps no expert: it names the rows of each split."""
        sizes = ", ".join(f"{name}={len(getattr(self.ctx, attr))}" for name, attr in (("fit", "df_fit"), ("dev", "df_dev"), ("val", "df_val")))
        return (
            f"No valid experts survived the fit (rows: {sizes}; cross-validation folds: {len(self.ctx.cv_folds)}). "
            "Every candidate formula failed to fit or to score on these rows: a formula needs more fit rows than it has coefficients, "
            "and dev and val need rows of their own. Give a longer history, fewer features, or pass df_train= so that AutoTAM sizes the splits from the data."
        )
#: </auto_tam_no_expert>

#: <auto_tam_history>
    @staticmethod
    def _training_rows(*frames: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """The rows given to ``fit()`` as they were given, in one frame (``None`` when there are none)."""
        given = [frame for frame in frames if frame is not None and len(frame)]
        return pd.concat(given, ignore_index=True) if given else None

    def _with_history(self, df_test: pd.DataFrame, date_col: Optional[str]) -> Tuple[pd.DataFrame, int]:
        """Puts the training rows in front of a frame that starts after them.

        The adaptive and Kalman experts rebuild their windows and their state from the frame they are given, so the
        forecast of a row depends on the rows that precede it. A frame gets the training rows that precede its first
        date in front of it, whether it starts after the training data or inside it (a frame that starts with the
        first training row gets nothing).

        Args:
            df_test: Frame to forecast.
            date_col: Name of the date column (the one of the fit when ``None``).

        Returns:
            The frame to run the experts on, and the number of rows that were added in front of ``df_test``.
        """
        date_col = date_col or getattr(self.ctx, "date_col", None)
        history = self.history_
        if history is None or not date_col or date_col not in df_test.columns or date_col not in history.columns or df_test.empty:
            return df_test, 0
        earlier = history[pd.to_datetime(history[date_col]) < pd.to_datetime(df_test[date_col]).min()]
        if earlier.empty:
            return df_test, 0
        columns = [c for c in df_test.columns if c in earlier.columns]
        missing = [c for c in earlier.columns if c not in df_test.columns]
        front = earlier[columns + [c for c in missing if c == self.ctx.target]]
        return pd.concat([front, df_test], ignore_index=True), len(front)
#: </auto_tam_history>

#: <auto_tam_predict>
    def _expert_predictions(self, df_test: pd.DataFrame, date_col: Optional[str] = None, external_experts: Optional[Dict[str, Any]] = None):
        """Run every trained expert on df_test -> (expert prediction frame, df_test_clean).

        Shared by predict() and predict_online() so the dynamic-expert row alignment
        (_align_prediction_to_test) lives in exactly one place.
        """
        if not self.trained_experts:
            raise ValueError("Model not fitted. No valid experts survived the fit process.")

        if hasattr(self.ctx, "external_mandatory_features") and self.ctx.external_mandatory_features:
            missing_ext = set(self.ctx.external_mandatory_features) - set(df_test.columns)
            if missing_ext:
                raise ValueError(
                    f"Test data is missing required external mandatory features: {sorted(missing_ext)}"
                )

        n_rows = len(df_test)
        external = normalise_external(external_experts, self.ctx.date_col or date_col, self.ctx.group_col)
        self._native_quantiles_ = {}
        df_test, n_history = self._with_history(df_test, date_col)
        df_test_clean, df_aug = self.data_manager.transform_test_data(df_test, self.ctx)
        predictions = {}
        
        for exp in self.trained_experts:
            m, name, m_type = exp["model"], exp["name"], exp["type"]
            if m_type == "external":
                if name in external:
                    predictions[name] = look_up(df_test_clean, external[name], self.ctx.date_col)
                continue
            try:
                base_m = m if m_type == "static" else getattr(m, '_saved_base_model', getattr(m, 'base_model_', None))
                f_ref = getattr(base_m, 'formula_', '')
                # The target names the formula but is not a feature: a forecast row has none, and must not be dropped for it.
                req_cols = sorted(set(c for c in formula_columns(f_ref, df_aug.columns) if c != self.ctx.target))
                df_aug_clean = df_aug.dropna(subset=req_cols).copy()
                
                if m_type == "static": 
                    static_out = base_m.predict(df_test_clean)
                    predictions[name] = aligned_values(static_out, df_test_clean, f"Estimated{self.ctx.target}", self.ctx.date_col, self.ctx.group_col)
                    if self._native_levels and hasattr(base_m, "predict_quantiles"):
                        self._native_quantiles_[name] = base_m.predict_quantiles(df_test_clean, self._native_levels).to_numpy(dtype=float)
                
                elif m_type == "kalman":
                    k_priors = getattr(self.ctx, 'kalman_priors', {})
                    obs_noise = k_priors.get("observation_noise_var", 0.1)
                    p_init = k_priors.get("P_init_diag", 1.0)
                    boost = k_priors.get("offset_boost", 100.0)

                    h_steps = getattr(self.ctx, 'inferred_horizon', len(df_test))
                    
                    fresh_kalman = KalmanTAM(
                        base_model=base_m, 
                        kalman_formula=exp["dynamic_formula"], 
                        date_col=self.ctx.date_col,
                        horizon_steps=h_steps, 
                        offset_boost=boost, 
                        observation_noise_var=obs_noise, 
                        P_init_diag=p_init,
                        process_noise_var=exp["params"]["process_noise_var"] 
                    )
                    # The filter follows the rows whose target is observed; the rows after the last observation are forecast with its final state.
                    kalman_col = f"KalmanAdapted_{self.ctx.target}"
                    observed_rows, forecast_rows = split_forecast_rows(df_aug_clean, self.ctx.target, self.ctx.group_col)
                    p_all = fresh_kalman.predict_online(observed_rows)
                    raw_preds = pd.Series(p_all[kalman_col].values, index=observed_rows.index)
                    if len(forecast_rows):
                        p_ahead = fresh_kalman.predict(forecast_rows)
                        raw_preds = pd.concat([raw_preds, pd.Series(p_ahead[kalman_col].values, index=forecast_rows.index)]).sort_index()
                    predictions[name] = self._align_prediction_to_test(raw_preds, df_aug_clean, df_test_clean)
                
                elif m_type == "adaptive":
                    h_steps = getattr(self.ctx, 'inferred_horizon', len(df_test))
                    s_per_period = getattr(self.ctx, 'steps_per_period', 1)
                    
                    fresh_adapt = AdaptiveTAM(
                        base_model=base_m, adaptive_formula=exp["dynamic_formula"], update_interval_periods=1,
                        training_window_periods=exp["params"]["training_window_periods"], 
                        steps_per_period=s_per_period, horizon_steps=h_steps
                    )
                    d_adapt = self.expander._prepare_meta_learning_data(df_aug_clean, base_m, self.ctx)
                    lag_cols = [c for c in d_adapt.columns if f"Residual{self.ctx.target}_lag_" in c]
                    d_adapt = d_adapt.dropna(subset=lag_cols).copy()
                    
                    if hasattr(fresh_adapt, 'predict_online'): 
                        fresh_adapt.predict_online(data=d_adapt)
                    else:
                        fresh_adapt.prepare_simulation(d_adapt)
                        fresh_adapt.simulation()
                        
                    raw_preds = fresh_adapt.predictions_[f"AdaptedEstimated{self.ctx.target}"]
                    predictions[name] = self._align_prediction_to_test(raw_preds, d_adapt, df_test_clean)
                    
            except Exception as e: 
                logger.warning(f"Prediction failed for {name}: {e}")
                
        preds_df = pd.DataFrame(predictions, index=df_test_clean.index)

        # Physical guardrail: Clip predictions to observed bounds (with margin) to prevent extreme OOD extrapolations from breaking static frozen-weight ensembles.
        bounds = getattr(self.ctx, "target_clip", None)
        if bounds is None and getattr(self.ctx, "df_fit", None) is not None:
            try:
                low = float(self.ctx.df_fit[self.ctx.target].min())
                high = float(self.ctx.df_fit[self.ctx.target].max())
                margin = 0.15 * (high - low)
                bounds = (low - margin, high + margin)
            except Exception as exc:
                log_failure(logger, "Training bounds of the target for the clipping of the experts", exc)
                bounds = None
        if bounds is not None and not preds_df.empty:
            clipped = preds_df.clip(lower=bounds[0], upper=bounds[1])
            if self.extrapolation_guards:
                counts = (clipped != preds_df).sum()
                self.clipped_rows_ = {str(name): int(n) for name, n in counts.items() if n}
                if self.clipped_rows_:
                    logger.info("Extrapolation guard: %d experts clipped (%d rows in total)", len(self.clipped_rows_), sum(self.clipped_rows_.values()))
            preds_df = clipped

        for alias, real_name in self.island_aliases.items():
            if real_name in preds_df.columns:
                preds_df[alias] = preds_df[real_name]

        if n_history:
            # The training rows were only there for the experts: the output has exactly the rows that were asked for.
            preds_df = preds_df.iloc[-n_rows:].reset_index(drop=True)
            df_test_clean = df_test_clean.iloc[-n_rows:].reset_index(drop=True)
            self._native_quantiles_ = {name: values[-n_rows:] for name, values in self._native_quantiles_.items()}

        self._note_complete(preds_df)
        return preds_df, df_test_clean

    def predict(self, df_test: pd.DataFrame, date_col: Optional[str] = None, by_identifier: bool = False,
                external_experts: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Frozen-weight inference (production path).

        League/apex weights learned on D_val during fit are applied unchanged across the whole
        horizon. Deterministic and target-free. For a chronological backtest where the realized
        target IS available, prefer predict_online(): MLpol then re-weights the experts
        sequentially, which matters a great deal across regime shifts.

        Args:
            df_test: Frame to forecast, with the features of the fit. It may hold the forecast rows alone, with a
                missing target: the training rows kept by ``fit()`` (``history_``) that precede its first date are put in
                front of it for the adaptive and Kalman experts, and the output has exactly the rows given, with every
                expert and ensemble complete up to the horizon (the smallest lag of the target). An ensemble mixes the
                experts that are finite on a row.
            date_col: Name of the date column, when the frame has one.
            by_identifier: Name the expert columns by their stable identifier (``experts_['identifier']``) instead of their name. The
                ensembles and the final model keep their names.
            external_experts: The forecasts of the external experts given to ``fit()``, on the rows of ``df_test`` (same forms). An external
                expert not given here is left out and the final model mixes the others.

        Returns:
            One row per row of ``df_test`` and one column per model, in this order: the experts, in search
            order; the league ensembles (``Ensemble_<league>``); the **final model**, in the column named by
            ``final_model_column_`` (``AutoTAM_Apex_Ensemble``). The first column is an expert, not the final
            model: read the final model through ``final_model_column_``.
        """
        df_test = self._with_derived_group(df_test)
        if self.difference_ and not self._differenced_input:
            return self._named(self._predict_levels(df_test, date_col), by_identifier)
        preds_df, df_test_clean = self._expert_predictions(df_test, date_col, external_experts)

        for league_name, weights in self.league_weights.items():
            if weights:
                valid_weights = {m: w for m, w in weights.items() if m in preds_df}
                if sum(valid_weights.values()) > 0:
                    final_name = league_name if league_name.startswith("Ensemble_") else f"Ensemble_{league_name}"
                    preds_df[final_name], left_out = finite_weighted_mix(preds_df, valid_weights)
                    if left_out:
                        logger.info("%s: %d expert values missing on some rows were left out and the weights renormalised", final_name, left_out)

        if self.weights_top10:
            valid_apex = {e: w for e, w in self.weights_top10.items() if e in preds_df}
            if sum(valid_apex.values()) > 0:
                preds_df[self.FINAL_MODEL_COLUMN], left_out = finite_weighted_mix(preds_df, valid_apex)
                if left_out:
                    logger.info("%s: %d expert values missing on some rows were left out and the weights renormalised", self.FINAL_MODEL_COLUMN, left_out)

        preds_df = self._guard_columns(preds_df, df_test_clean)
        preds_df = self._gated(preds_df, df_test if len(df_test) == len(preds_df) else df_test_clean)   # the frame as given still has the closed columns

        target = self.ctx.target
        if target in df_test.columns and df_test[target].notna().all():
            print("Hint: df_test carries a realized target, so this is a backtest. predict() applies "
                  "FROZEN league/apex weights learned on D_val; call predict_online() to let MLpol "
                  "re-weight the experts chronologically across regime shifts.")

        return self._named(preds_df, by_identifier)

    def _named(self, preds_df: pd.DataFrame, by_identifier: bool) -> pd.DataFrame:
        """The columns of the experts renamed by identifier when asked."""
        if not by_identifier:
            return preds_df
        return preds_df.rename(columns=identifiers(self.trained_experts))

    def _note_complete(self, preds_df: pd.DataFrame) -> None:
        """Records in ``experts_`` which experts were finite on every row of the last prediction."""
        if self.experts_ is None or self.experts_.empty:
            return
        flags = {name: bool(np.isfinite(pd.to_numeric(preds_df[name], errors="coerce").to_numpy()).all()) if name in preds_df else False
                 for name in self.experts_["name"]}
        self.experts_["complete"] = self.experts_["name"].map(flags)

#: <auto_tam_quantiles>
    def predict_quantiles(self, df_test: pd.DataFrame, taus: Sequence[float] = (0.1, 0.5, 0.9), date_col: Optional[str] = None,
                          aggregation: str = "empirical") -> pd.DataFrame:
        """Quantile forecasts of the final model: its point forecast plus the quantiles of its validation errors.

        The error of the final model on the validation rows (``y - forecast``, rows the search did not fit) is the empirical law of the
        error; the quantile of level ``tau`` is the point forecast plus the ``tau``-quantile of that error, taken in the group of the row
        when the group has at least 30 validation errors, else over all groups. Levels are returned sorted, so the columns never cross. The
        coverage is the one of the validation errors: it holds when the future errors look like them (the error of a longer horizon is
        not wider here).

        Args:
            df_test: Frame to forecast, as for ``predict()``.
            taus: Quantile levels in (0, 1).
            date_col: Name of the date column, when the frame has one.
            aggregation: ``"empirical"`` (the default) as described above. ``"opera"`` aggregates, level by level, the quantile
                forecasts of the members of the final model: each member forecasts the ``tau``-quantile as its own point forecast plus
                the ``tau``-quantile of its own validation errors, and one ``OperaTAM`` with the pinball loss of that level, run over
                the validation rows, gives the weights (the last ones are applied to the rows to forecast). Levels are sorted on each
                row. The quantile of a member comes from all the validation errors of the member, not from its group. With a single
                member it is the ``"empirical"`` forecast.

        Returns:
            One row per row of ``df_test`` and one ``q<tau>`` column per level, in increasing order of level.

        Raises:
            ValueError: A level is outside (0, 1), ``aggregation`` is unknown, or the model has no final model (it needs the Apex
                ensemble: ``use_opera=True``).
            NotImplementedError: ``aggregation="opera"`` on a model fitted on increments (``difference``).
        """
        if aggregation not in ("empirical", "opera"):
            raise ValueError(f"aggregation must be 'empirical' or 'opera'; got {aggregation!r}")
        if aggregation == "opera" and self.difference_:
            raise NotImplementedError("predict_quantiles(aggregation='opera') is not available for a model fitted on increments (difference=...).")
        levels = sorted(float(tau) for tau in taus)
        if not levels or any(not 0.0 < tau < 1.0 for tau in levels):
            raise ValueError(f"taus must lie in (0, 1); got {list(taus)!r}")
        column = self.final_model_column_
        if column is None or column not in self.oof_predictions_:
            raise ValueError("predict_quantiles needs the final model of the Apex ensemble: fit with use_opera=True and enough rows for a validation part.")
        df_test = self._with_derived_group(df_test)
        self._native_levels = levels if aggregation == "opera" and self.native_quantiles else None
        try:
            predictions = self.predict(df_test, date_col)
        finally:
            self._native_levels = None
        point = predictions[column]
        val = self.ctx.df_val
        in_val = np.asarray(self.oof_predictions_[column], dtype=float)[-len(val):]   # the validation rows come last in the fit + dev + val frame
        errors = (val[self.ctx.target] - pd.Series(in_val, index=val.index)).dropna()
        if errors.empty:
            raise ValueError("predict_quantiles: the validation part has no usable error.")
        if aggregation == "opera":
            aggregated = self._opera_quantiles(predictions, df_test, levels, date_col)
            if aggregated is not None:
                return self._gated(pd.DataFrame(aggregated, index=point.index, columns=[f"q{tau}" for tau in levels]), df_test)
        pooled = np.quantile(errors.to_numpy(dtype=float), levels)
        out = {f"q{tau}": point.to_numpy(dtype=float) + pooled[i] for i, tau in enumerate(levels)}
        group_col = self.ctx.group_col
        if group_col and group_col in val.columns and group_col in df_test.columns:
            labels = df_test[group_col].to_numpy()
            for key, group_errors in errors.groupby(val.loc[errors.index, group_col]):
                if len(group_errors) >= 30:
                    own = np.quantile(group_errors.to_numpy(dtype=float), levels)
                    rows = labels == key
                    for i, tau in enumerate(levels):
                        out[f"q{tau}"][rows] = point.to_numpy(dtype=float)[rows] + own[i]
        return self._gated(pd.DataFrame(out, index=point.index), df_test)

    def _opera_quantiles(self, predictions: pd.DataFrame, df_test: pd.DataFrame, levels: List[float],
                         date_col: Optional[str]) -> Optional[np.ndarray]:
        """Per-level Opera of the quantile forecasts of the members of the final model, and of the native quantile experts when there are some.

        A member of the final model forecasts a level as its point forecast plus the quantile of its own validation errors; a native expert
        (``native_quantiles``) with its own quantiles. The pinball loss of each level weights them, so a native expert that is worse than the
        members gets a small weight; the point forecast of the final model is not changed by them. ``quantile_members_`` lists the members used.

        Returns:
            An array of one row per row of ``df_test`` and one column per level, sorted on each row; ``None`` with fewer than two members.
        """
        val, target = self.ctx.df_val, self.ctx.target
        by_name = {e["name"]: e["model"] for e in self.trained_experts}
        points = [m for m in self.weights_top10 if m in self.oof_predictions_ and m in predictions.columns]
        natives = [m for m in self._native_quantiles_ if m in by_name and m not in points and hasattr(by_name[m], "predict_quantiles")]
        if len(points) + len(natives) < 2:
            return None
        d_col = date_col or self.ctx.date_col
        g_col = self.ctx.group_col
        truth = pd.to_numeric(val[target], errors="coerce").to_numpy(dtype=float)
        history_columns: Dict[str, np.ndarray] = {}      # {member: array rows x levels} on the validation rows
        future_columns: Dict[str, np.ndarray] = {}       # the same on the rows to forecast
        for member in points:
            in_val = np.asarray(self.oof_predictions_[member], dtype=float)[-len(val):]
            errors = truth - in_val
            offsets = np.quantile(errors[np.isfinite(errors)], levels)
            history_columns[member] = in_val[:, None] + offsets[None, :]
            future_columns[member] = predictions[member].to_numpy(dtype=float)[:, None] + offsets[None, :]
        for member in natives:
            try:   # the expert's own quantiles on the validation rows (NaN where a feature is missing)
                required = [c for c in formula_columns(getattr(by_name[member], "formula_", ""), val.columns) if c != target]
                own = by_name[member].predict_quantiles(val.dropna(subset=required), levels).reindex(val.index)
                history_columns[member] = own.to_numpy(dtype=float)
                future_columns[member] = self._native_quantiles_[member]
            except Exception as exc:
                log_failure(logger, f"Native quantiles of {member} on the validation rows", exc)
        members = list(history_columns)
        if len(members) < 2:
            return None
        self.quantile_members_ = members
        keep = np.isfinite(truth)
        for values in history_columns.values():
            keep &= np.isfinite(values).all(axis=1)
        if keep.sum() < 2:
            return None
        names = {member: f"member_{j}" for j, member in enumerate(members)}
        carried = [c for c in (d_col, g_col) if c and c in val.columns and c in df_test.columns]
        columns = {c: val[c].to_numpy()[keep] for c in carried}
        test_columns = {c: df_test[c].to_numpy() for c in carried}
        table = np.empty((len(df_test), len(levels)))
        for i, tau in enumerate(levels):
            history = pd.DataFrame({**columns, target: truth[keep]})
            future = pd.DataFrame(test_columns, index=range(len(df_test)))
            for member in members:
                history[names[member]] = history_columns[member][keep, i]
                future[names[member]] = future_columns[member][:, i]
            opera = OperaTAM(target_col=target, expert_cols=list(names.values()), algorithm="MLPOL", loss_type="pinball", tau=tau,
                             date_col=d_col if d_col in carried else None, group_col=g_col if g_col in carried else None)
            opera.fit(history)
            table[:, i] = opera.predict(future)["prediction_opera"].to_numpy(dtype=float)
        return np.sort(table, axis=1)
#: </auto_tam_quantiles>

    def _predict_levels(self, df_test: pd.DataFrame, date_col: Optional[str]) -> pd.DataFrame:
        """``predict()`` of a model fitted on increments: forecasts the increments, then adds the levels back (see ``differencing.py``)."""
        date_col = date_col or self.ctx.date_col
        target, group_col = self._difference_target, self.ctx.group_col
        increments = difference_test(df_test, self._levels, target, date_col, group_col, self.difference_)
        self._differenced_input = True
        try:
            forecast = self.predict(increments, date_col)
        finally:
            self._differenced_input = False
        return add_back(forecast, df_test, self._levels, target, date_col, group_col, self.difference_)

    def predict_online(self, df_test: pd.DataFrame, date_col: Optional[str] = None, by_identifier: bool = False,
                       external_experts: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Sequential inference: MLpol re-weights the experts chronologically.

        Mirrors OperaTAM.predict_online at the orchestrator level. Requires the realized target in
        df_test, because online aggregation scores every expert's loss at each step to update its
        weight; it is therefore a backtest/replay path, not a target-free forward forecast.

        Adds 'AutoTAM_Apex_Online' and 'Ensemble_<league>_Online' next to the individual experts,
        and stores the weight trajectories on self.online_weights_ for diagnostics/reporting.
        The online Apex aggregates the whole Apex pool (apex_members_); predict() averages only the
        members still weighted at the end of validation (weights_top10).

        Args:
            df_test: Frame to forecast, with the features of the fit and the realized target. As for ``predict()``, a
                frame that starts after the training data gets the training rows in front of it internally.
            date_col: Name of the date column, when the frame has one.
            by_identifier: As for ``predict()``.
            external_experts: As for ``predict()``; the online Apex aggregates the ones given.

        Returns:
            The same columns as ``predict()`` (experts, then ``Ensemble_<league>``) plus the online versions
            ``Ensemble_<league>_Online`` and the online **final model** in ``FINAL_MODEL_COLUMN_ONLINE``
            (``AutoTAM_Apex_Online``).

        Raises:
            NotImplementedError: The model learns increments (``difference``): use ``predict()``.
        """
        df_test = self._with_derived_group(df_test)
        if self.difference_:
            raise NotImplementedError("predict_online() is not available for a model fitted on increments (difference=...): use predict().")
        preds_df, df_test_clean = self._expert_predictions(df_test, date_col, external_experts)
        target = self.ctx.target

        y_series = pd.to_numeric(df_test_clean[target], errors="coerce") if target in df_test_clean.columns else None
        if y_series is None or y_series.isna().any():
            raise ValueError(
                f"predict_online() requires the realized target column '{target}' (fully populated) "
                "in df_test: MLpol updates each expert's weight from its observed loss at every "
                "step. Use predict() for target-free forward inference."
            )

        d_col, g_col = self.ctx.date_col, self.ctx.group_col
        base = pd.DataFrame(index=df_test_clean.index)
        base[target] = y_series.to_numpy()
        if d_col and d_col in df_test_clean.columns:
            base[d_col] = pd.to_datetime(df_test_clean[d_col]).to_numpy()
        if g_col and g_col in df_test_clean.columns:
            base[g_col] = df_test_clean[g_col].to_numpy()

        self.online_weights_ = {}

        def _aggregate_online(members, out_name):
            usable = [m for m in members if m in preds_df.columns
                      and np.isfinite(pd.to_numeric(preds_df[m], errors="coerce").to_numpy()).all()]
            if len(usable) < 2:
                return
            frame = base.copy()
            for m in usable:
                frame[m] = pd.to_numeric(preds_df[m], errors="coerce").to_numpy()
            formula = f"{target} ~ " + " + ".join(f"l({m})" for m in usable)
            try:
                opera = OperaTAM(formula=formula, algorithm="MLPOL", date_col=d_col, group_col=g_col)
                result = opera.predict_online(frame)
                aggregated = pd.to_numeric(result["prediction_opera"], errors="coerce").to_numpy()
                if len(aggregated) != len(preds_df):
                    logger.warning(f"online aggregation for {out_name} returned {len(aggregated)} rows "
                          f"for {len(preds_df)} test rows; skipped.")
                    return
                preds_df[out_name] = aggregated
                weight_cols = [c for c in result.columns if c.startswith("weight_")]
                if weight_cols:
                    self.online_weights_[out_name] = result[weight_cols].reset_index(drop=True)
            except Exception as exc:
                logger.warning(f"online aggregation failed for {out_name}: {exc}")

        for league_name, weights in self.league_weights.items():
            if weights:
                stem = league_name if league_name.startswith("Ensemble_") else f"Ensemble_{league_name}"
                _aggregate_online(list(weights.keys()), f"{stem}_Online")

        # The online Apex dynamically aggregates the full pool; static predict() averages only the snapshot weights.
        apex_members = list(getattr(self, "apex_members_", None) or self.weights_top10.keys())
        if apex_members:
            _aggregate_online(apex_members, self.FINAL_MODEL_COLUMN_ONLINE)

        return self._named(self._gated(self._guard_columns(preds_df, df_test_clean), df_test if len(df_test) == len(preds_df) else df_test_clean), by_identifier)

    def _align_prediction_to_test(self, raw_preds, source_df, df_test_clean):
        """Realign dynamic expert predictions (augmented history + test) back
        to the original test rows using date/group keys.

        """
        n = len(df_test_clean)
        date_col, group_col = self.ctx.date_col, self.ctx.group_col
        try:
            if hasattr(raw_preds, "index") and raw_preds.index.isin(source_df.index).all():
                src_keys = source_df.loc[raw_preds.index]
                vals = np.asarray(raw_preds.values, dtype=float)
            elif len(raw_preds) == len(source_df):
                src_keys = source_df
                vals = np.asarray(getattr(raw_preds, "values", raw_preds), dtype=float)
            else:
                raise ValueError("raw_preds not alignable to source_df")

            if date_col and date_col in source_df.columns and date_col in df_test_clean.columns:
                gcol = group_col if (group_col and group_col in source_df.columns
                                     and group_col in df_test_clean.columns) else None
                keys = [date_col] + ([gcol] if gcol else [])
                freq = getattr(self.ctx, "metadata", {}).get("delta_t", "1D")
                src = src_keys[keys].copy()
                src["__v__"] = vals
                src[date_col] = pd.to_datetime(src[date_col])
                left = df_test_clean[keys].copy()
                left[date_col] = pd.to_datetime(left[date_col])
                try:
                    src[date_col] = src[date_col].dt.floor(freq)
                    left[date_col] = left[date_col].dt.floor(freq)
                except (ValueError, TypeError):
                    pass
                src = src.drop_duplicates(subset=keys, keep="last")
                left["__ord__"] = range(n)
                merged = left.merge(src, on=keys, how="left").sort_values("__ord__")
                out = merged["__v__"].to_numpy()
                if len(out) == n and np.isfinite(out).any():
                    return out
        except Exception as exc:
            log_failure(logger, "Alignment of the predictions on the key columns (falls back to positions)", exc)
        vals = np.asarray(getattr(raw_preds, "values", raw_preds), dtype=float)
        return vals[-n:] if len(vals) >= n else np.full(n, np.nan)
#: </auto_tam_predict>

#: <auto_tam_utils>
    def summary(self) -> None:
        self.reporter.print_summary(self.trained_experts, {}, {})
        if getattr(self, 'extrapolating_features_', None):
            print(f"\nFeatures extrapolated outside the trained range during the search: {', '.join(self.extrapolating_features_)}")

    def print_performance_board(self) -> None:
        print("\n" + "="*90 + "\nTAM: Global Model Performance Leaderboard\n" + "="*90)
        valid_logs = [log for log in self.chronological_test_log if pd.notna(log.get("Validation_RMSE"))]
        if not valid_logs: return
        
        valid_logs.sort(key=lambda x: float(x.get("Penalized_Score", x.get("Validation_RMSE"))))
        
        metric_display = getattr(self.ctx, 'optimization_metric', 'rmse').upper() if self.ctx else 'SCORE'
        
        for rank, log in enumerate(valid_logs, 1):
            raw_score = float(log.get('Validation_RMSE'))
            pen_score = float(log.get('Penalized_Score', raw_score))
            comp = log.get('Complexity', 'N/A')
            
            print(f"\n[{rank}] {log.get('Model_Name')} ({log.get('Model_Type')}) | Params: {comp} | Penalized {metric_display}: {pen_score:.5f} (Raw: {raw_score:.5f})\n    -> Formula: {log.get('Formula')}")
            params = log.get("Hyperparameters", "")
            if params and params not in ["{}", "Unpenalized_Ridge", "GCV_Auto_Penalized", "GCV_Auto"]: 
                print(f"    -> Params:  {params}")
        print("\n" + "="*90)

    def summary_report(self):
        """
        Convenience method to generate the report directly from the model object.
        """
        opt_metric = getattr(self.ctx, 'optimization_metric', 'RMSE') if self.ctx else 'RMSE'
        generate_autotam_report(export_path=self.export_directory, metric=opt_metric)

    def print_model_recipe(self, internal_model_name: str) -> None:
        """
        Prints the complete mathematical recipe for a given model or ensemble.
        Includes aggregation weights for OPERA models and base formulas for dynamic models.
        """
        print_model_recipe(self, internal_model_name)
#: </auto_tam_utils>
#: </auto_tam_class>