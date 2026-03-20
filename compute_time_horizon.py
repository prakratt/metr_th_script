#!/usr/bin/env python3
"""
Standalone script for computing METR Time Horizons.

This script replicates the core computation from METR's eval-analysis-public
repository (https://github.com/METR/eval-analysis-public) as a single file
for third-party replication and analysis.

A "time horizon" measures the task duration (in human-expert minutes) at which
an AI model achieves a given success rate. For example, a 50% time horizon of
60 minutes means the model succeeds ~50% of the time on tasks that take humans
60 minutes to complete.

Methodology:
  1. Weight tasks to balance representation across task families
  2. Fit logistic regression: log2(human_minutes) -> P(success)
  3. Extract time horizons at specified success percentiles (e.g., 50%, 80%)
  4. Bootstrap for confidence intervals (hierarchical: family -> task -> run)
  5. Optionally compute doubling time trend across model release dates

Reference: "Measuring the Capability Frontier" (arXiv:2503.14499)

Usage:
  python compute_time_horizon.py compute --input runs.jsonl --release-dates dates.yaml
  python compute_time_horizon.py convert --input my_data.csv --column-map model=agent,duration_hrs=human_minutes
  python compute_time_horizon.py selftest
  python compute_time_horizon.py validate --input runs.jsonl --expected results.yaml
  python compute_time_horizon.py plot --input results.yaml --release-dates dates.yaml

Dependencies: numpy, pandas, scikit-learn, pyyaml, joblib
Optional: matplotlib (for plotting)

Input Format:
  The script expects run-level data (one row per evaluation run) with columns:
    - task_id:        Unique task identifier (string)
    - task_family:    Group of related tasks (string). If unavailable, use task_id.
    - agent:          Model/agent name (string). Aliases: "alias", "model_name"
    - human_minutes:  Human expert completion time in minutes (float > 0)
    - score_binarized: Binary success score, 0 or 1 (or "score", "success", "pass")
    - run_id:         Unique run identifier (auto-generated if absent)

  Use `convert` subcommand or `--column-map` to adapt other formats.
  Use `--auto-family` to auto-generate task_family from task_id prefixes.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import pathlib
import sys
import warnings
from typing import Any

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression, LogisticRegression

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.dates import date2num, num2date

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_REGULARIZATION = 1e-5
DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_SEED = 42
DEFAULT_SUCCESS_PERCENTS = [50, 80]
DEFAULT_WEIGHTING = "invsqrt_task_weight"
DEFAULT_SCORE_COL = "score_binarized"
TIME_BUCKETS = [1, 4, 16, 64, 256, 960, 2880]

# ── Section 2: Data Loading & Validation ─────────────────────────────────────


def _read_file(filepath: str) -> pd.DataFrame:
    """Read CSV, JSONL, or JSON file into a DataFrame."""
    path = pathlib.Path(filepath)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True, orient="records", convert_dates=False)
    elif path.suffix == ".csv":
        return pd.read_csv(path)
    elif path.suffix == ".json":
        return pd.read_json(path, convert_dates=False)
    elif path.suffix in (".tsv",):
        return pd.read_csv(path, sep="\t")
    else:
        raise ValueError(f"Unsupported file format: {path.suffix} (use .jsonl, .csv, .json, .tsv)")


# Known aliases for each required column
_COLUMN_ALIASES: dict[str, list[str]] = {
    "agent": ["alias", "model_name", "model", "agent_name", "system"],
    "task_id": ["task", "problem_id", "question_id", "item_id"],
    "task_family": ["family", "category", "domain", "task_group", "suite"],
    "human_minutes": ["human_time", "duration_minutes", "time_minutes",
                      "human_duration", "baseline_minutes", "expert_minutes"],
    "score_binarized": ["score", "success", "pass", "correct", "solved",
                        "score_binary", "passed", "result"],
    "run_id": ["run", "attempt_id", "trial_id", "evaluation_id", "attempt"],
}


def _apply_column_mapping(
    df: pd.DataFrame,
    column_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Apply explicit column mapping, then try known aliases for missing columns.

    Args:
        df: Input DataFrame.
        column_map: Explicit mapping of {source_col: target_col} or
                    {target_col: source_col} (both directions are tried).

    Returns:
        DataFrame with standardized column names.
    """
    df = df.copy()

    # Apply explicit mapping
    if column_map:
        rename = {}
        for k, v in column_map.items():
            # Support both directions: source=target and target=source
            if k in df.columns:
                rename[k] = v
            elif v in df.columns:
                rename[v] = k
        df = df.rename(columns=rename)

    # Try known aliases for any still-missing columns
    for target, aliases in _COLUMN_ALIASES.items():
        if target not in df.columns:
            for alias in aliases:
                if alias in df.columns:
                    df = df.rename(columns={alias: target})
                    break

    return df


def _auto_generate_families(df: pd.DataFrame) -> pd.DataFrame:
    """Auto-generate task_family from task_id prefixes.

    Splits task_id on common separators (/, _, -) and uses the first component.
    If all tasks end up in one family, uses each task as its own family.
    """
    df = df.copy()
    # Try splitting on common separators
    for sep in ["/", "_", "-", "."]:
        families = df["task_id"].astype(str).str.split(sep).str[0]
        if families.nunique() > 1:
            df["task_family"] = families
            return df
    # Fallback: each task is its own family
    df["task_family"] = df["task_id"]
    return df


def _convert_time_units(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and convert time columns that aren't in minutes.

    Checks for columns with hours/seconds suffixes and converts to minutes.
    """
    df = df.copy()
    for col_suffix, factor in [("_hours", 60), ("_hrs", 60), ("_seconds", 1/60), ("_secs", 1/60)]:
        for col in list(df.columns):
            if col.endswith(col_suffix) and "human_minutes" not in df.columns:
                target = col.replace(col_suffix, "_minutes")
                df[target] = df[col] * factor
                # Try to map to human_minutes
                if target in ("human_minutes", "duration_minutes", "time_minutes"):
                    df["human_minutes"] = df[target]
    # Also check for explicitly named columns
    if "human_minutes" not in df.columns:
        if "human_hours" in df.columns:
            df["human_minutes"] = df["human_hours"] * 60
        elif "human_seconds" in df.columns:
            df["human_minutes"] = df["human_seconds"] / 60
        elif "duration_hours" in df.columns:
            df["human_minutes"] = df["duration_hours"] * 60
    return df


def load_run_data(
    filepath: str,
    score_col: str = DEFAULT_SCORE_COL,
    column_map: dict[str, str] | None = None,
    auto_family: bool = False,
) -> pd.DataFrame:
    """Load run-level evaluation data from JSONL, CSV, JSON, or TSV.

    Flexible loader that accepts various column naming conventions:
      - agent: also accepts "alias", "model_name", "model", "system"
      - task_id: also accepts "task", "problem_id", "question_id"
      - task_family: also accepts "family", "category", "domain", "suite"
      - human_minutes: also accepts "human_time", "duration_minutes", "human_hours" (auto-converted)
      - score_binarized: also accepts "score", "success", "pass", "correct", "solved"
      - run_id: auto-generated if absent

    Args:
        filepath: Path to data file (.jsonl, .csv, .json, .tsv).
        score_col: Name of the score column (before alias resolution).
        column_map: Explicit column mapping, e.g. {"my_model_col": "agent"}.
        auto_family: If True and task_family is missing, auto-generate from task_id.

    Returns:
        DataFrame with standardized column names.

    Example:
        # Your data has "model", "problem", "duration_hours", "correct":
        df = load_run_data("my_eval.csv",
                           column_map={"problem": "task_id", "duration_hours": "human_hours"},
                           score_col="correct", auto_family=True)
    """
    df = _read_file(filepath)

    # Apply column mapping and aliases
    df = _apply_column_mapping(df, column_map)

    # Handle time unit conversions
    df = _convert_time_units(df)

    # Auto-generate task_family if missing and requested
    if "task_family" not in df.columns:
        if auto_family:
            df = _auto_generate_families(df)
        else:
            raise ValueError(
                "Column 'task_family' not found. Use --auto-family to generate from task_id, "
                "or provide a mapping with --column-map family_col=task_family"
            )

    # Resolve score_col: if it was mapped to score_binarized already, use that;
    # otherwise try to find score_col or its aliases and rename to score_col
    if score_col not in df.columns:
        if score_col in ("score_binarized",) and "score_binarized" in df.columns:
            pass  # Already resolved
        else:
            # Check if score_col was renamed to score_binarized by aliases
            if "score_binarized" in df.columns and score_col != "score_binarized":
                # The user's score_col was resolved by alias mapping
                score_col = "score_binarized"
            else:
                for alias in _COLUMN_ALIASES.get("score_binarized", []):
                    if alias in df.columns:
                        df = df.rename(columns={alias: score_col})
                        break

    # Validate required columns
    required = ["task_id", "task_family", "agent", "human_minutes", score_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        available = sorted(df.columns.tolist())
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {available}\n"
            f"Hint: use --column-map to map your columns, e.g.: --column-map '{missing[0]}=your_col_name'"
        )

    # Generate run_id if absent
    if "run_id" not in df.columns:
        df["run_id"] = [f"run_{i}" for i in range(len(df))]

    # Validate data
    if not (df[score_col] >= 0).all() or not (df[score_col] <= 1).all():
        raise ValueError(f"Score column '{score_col}' must have values in [0, 1]")
    if not (df["human_minutes"] > 0).all():
        raise ValueError("human_minutes must be positive")
    for col in required:
        if df[col].isna().any():
            raise ValueError(f"Column '{col}' contains null values")

    return df


def load_release_dates(filepath: str) -> dict[str, str]:
    """Load release dates from YAML file.

    Handles both formats:
      - {date: {agent: "YYYY-MM-DD", ...}}
      - {agent: "YYYY-MM-DD", ...}

    Returns:
        Dict mapping agent name to date string.
    """
    data = yaml.safe_load(pathlib.Path(filepath).read_text())
    if "date" in data and isinstance(data["date"], dict):
        return {k: str(v) for k, v in data["date"].items()}
    return {k: str(v) for k, v in data.items()}


# ── Section 3: Task Weight Computation ───────────────────────────────────────


def compute_task_weights(
    df_agent: pd.DataFrame, scheme: str = DEFAULT_WEIGHTING
) -> pd.Series:
    """Compute per-run sample weights for a single agent's data.

    Two schemes:
      - equal_task_weight: 1/n_runs_in_task, normalized to sum=1
      - invsqrt_task_weight: (1/n_runs_in_task) * (1/sqrt(n_tasks_in_family)), normalized

    Args:
        df_agent: DataFrame for a single agent with columns [task_id, task_family].
        scheme: Weighting scheme name.

    Returns:
        Series of weights, normalized to sum to 1.0, indexed same as df_agent.
    """
    # Count runs per task
    task_run_counts = df_agent.groupby("task_id")["run_id"].count()
    equal_weight = 1.0 / df_agent["task_id"].map(task_run_counts)

    if scheme == "equal_task_weight":
        weights = equal_weight / equal_weight.sum()
    elif scheme == "invsqrt_task_weight":
        # Count tasks per family
        task_families = (
            df_agent.groupby("task_id")["task_family"].first()
        )
        family_sizes = task_families.reset_index().groupby("task_family")["task_id"].count()
        invsqrt_family = 1.0 / np.sqrt(df_agent["task_family"].map(family_sizes))
        weights = equal_weight * invsqrt_family
        weights = weights / weights.sum()
    else:
        raise ValueError(f"Unknown weighting scheme: {scheme}")

    assert np.allclose(weights.sum(), 1.0), f"Weights sum to {weights.sum()}, expected 1.0"
    return weights.rename("weight")


def add_weight_column(df: pd.DataFrame, scheme: str = DEFAULT_WEIGHTING) -> pd.DataFrame:
    """Add a 'weight' column to the DataFrame, computed per agent.

    Args:
        df: Full DataFrame with all agents.
        scheme: Weighting scheme.

    Returns:
        DataFrame with added 'weight' column.
    """
    weight_series = pd.concat(
        [compute_task_weights(agent_df, scheme) for _, agent_df in df.groupby("agent")],
        ignore_index=False,
    )
    df = df.copy()
    df["weight"] = weight_series
    return df


# ── Section 4: Core Logistic Regression ──────────────────────────────────────


def fit_logistic(
    X: NDArray[Any],
    y: NDArray[Any],
    sample_weight: NDArray[Any],
    regularization: float = DEFAULT_REGULARIZATION,
    ensure_weights_sum_to_1: bool = True,
) -> LogisticRegression:
    """Fit weighted logistic regression, handling fractional y values.

    For fractional y (0 < y < 1), each observation is split into two weighted
    binary observations: one y=0 with weight (1-y)*w, one y=1 with weight y*w.
    This preserves the weighted average of the original data.

    Args:
        X: Feature array, shape (n, 1). Typically log2(human_minutes).
        y: Target array, values in [0, 1].
        sample_weight: Per-sample weights.
        regularization: L2 regularization strength (C = 1/regularization).
        ensure_weights_sum_to_1: Assert weights sum to 1.0 if True.

    Returns:
        Fitted sklearn LogisticRegression model.
    """
    assert np.all((y >= 0) & (y <= 1)), "y values must be in [0, 1]"

    if ensure_weights_sum_to_1:
        assert np.allclose(np.sum(sample_weight), 1.0), (
            f"sample_weight must sum to 1.0, got {np.sum(sample_weight)}"
        )

    original_weight_sum = np.sum(sample_weight)
    original_average = np.average(y, weights=sample_weight)

    # Split fractional y values into weighted binary observations
    fractional_mask = (y > 0) & (y < 1)
    if np.any(fractional_mask):
        X_frac = X[fractional_mask]
        y_frac = y[fractional_mask]
        w_frac = sample_weight[fractional_mask]

        X_split = np.vstack([X_frac, X_frac])
        y_split = np.zeros(2 * len(y_frac))
        y_split[len(y_frac):] = 1
        w_split = np.concatenate([(1 - y_frac) * w_frac, y_frac * w_frac])

        X = np.vstack([X[~fractional_mask], X_split])
        y = np.concatenate([y[~fractional_mask], y_split])
        sample_weight = np.concatenate([sample_weight[~fractional_mask], w_split])

        assert np.allclose(np.sum(sample_weight), original_weight_sum)
        assert np.allclose(np.average(y, weights=sample_weight), original_average)

    model = LogisticRegression(C=1.0 / regularization)
    model.fit(X, y, sample_weight=sample_weight)
    return model


def get_horizon_at_percent(model: LogisticRegression, success_percent: int) -> float:
    """Extract time horizon in minutes from a fitted logistic model.

    Solves: P(success) = success_percent/100 for minutes.
    P(success) = sigmoid(coefficient * log2(minutes) + intercept)

    Args:
        model: Fitted logistic regression model.
        success_percent: Target success rate (e.g., 50 for p50).

    Returns:
        Time horizon in minutes (2^x where x is the log2-minutes solution).
    """
    q = success_percent / 100.0
    log_odds = np.log(q / (1 - q))
    x = (log_odds - model.intercept_[0]) / model.coef_[0][0]
    return float(np.exp2(x))


# ── Section 5: Per-Agent Regression ──────────────────────────────────────────


def compute_agent_horizon(
    human_minutes: NDArray[Any],
    scores: NDArray[Any],
    weights: NDArray[Any],
    agent_name: str,
    regularization: float = DEFAULT_REGULARIZATION,
    success_percents: list[int] | None = None,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    bootstrap_horizons: pd.DataFrame | None = None,
    ensure_weights_sum_to_1: bool = True,
) -> dict[str, Any]:
    """Compute time horizon metrics for a single agent.

    Args:
        human_minutes: Array of task durations in minutes.
        scores: Array of success scores in [0, 1].
        weights: Per-run sample weights.
        agent_name: Name of the agent/model.
        regularization: L2 regularization parameter.
        success_percents: List of success percentiles (e.g., [50, 80]).
        confidence_level: Confidence level for CIs (e.g., 0.95).
        bootstrap_horizons: DataFrame of bootstrap results for CI extraction.
        ensure_weights_sum_to_1: Whether to assert weight normalization.

    Returns:
        Dict with keys: agent, coefficient, intercept, average_score,
        and for each percent p: p{p}, p{p}_ci_low, p{p}_ci_high.
    """
    if success_percents is None:
        success_percents = DEFAULT_SUCCESS_PERCENTS

    X = np.log2(human_minutes).reshape(-1, 1)
    average_score = float(np.average(scores, weights=weights))

    low_q = (1 - confidence_level) / 2
    high_q = 1 - low_q

    result: dict[str, Any] = {"agent": agent_name, "average_score": average_score}

    # Handle edge case: all scores are 0
    if np.all(scores == 0):
        result["coefficient"] = float("-inf")
        result["intercept"] = 0.0
        for p in success_percents:
            result[f"p{p}"] = 0.0
            result[f"p{p}_ci_low"] = 0.0
            result[f"p{p}_ci_high"] = 0.0
        return result

    model = fit_logistic(
        X, scores, sample_weight=weights,
        regularization=regularization,
        ensure_weights_sum_to_1=ensure_weights_sum_to_1,
    )

    if model.coef_[0][0] > 0:
        logger.warning(f"{agent_name} has positive slope {model.coef_[0][0]}")

    result["coefficient"] = float(model.coef_[0][0])
    result["intercept"] = float(model.intercept_[0])

    for p in success_percents:
        horizon = get_horizon_at_percent(model, p)
        result[f"p{p}"] = horizon

        if (
            bootstrap_horizons is not None
            and f"{agent_name}_p{p}" in bootstrap_horizons.columns
        ):
            result[f"p{p}_ci_low"] = float(
                np.nanquantile(bootstrap_horizons[f"{agent_name}_p{p}"], low_q)
            )
            result[f"p{p}_ci_high"] = float(
                np.nanquantile(bootstrap_horizons[f"{agent_name}_p{p}"], high_q)
            )
        else:
            result[f"p{p}_ci_low"] = float("nan")
            result[f"p{p}_ci_high"] = float("nan")

    return result


# ── Section 6: Bootstrap Confidence Intervals ────────────────────────────────


def bootstrap_runs_by_task_agent(
    task_col: NDArray[Any],
    agent_col: NDArray[Any],
    indices: NDArray[Any],
    rng: np.random.Generator,
) -> NDArray[Any]:
    """Bootstrap runs within each (task, agent) group.

    Args:
        task_col: Array of task IDs (full dataset).
        agent_col: Array of agent names (full dataset).
        indices: Current indices to sample from.
        rng: Random number generator.

    Returns:
        Array of resampled indices.
    """
    task_agent = np.char.add(
        np.char.add(task_col.astype(str), "|||"), agent_col.astype(str)
    )
    task_agents = task_agent[indices]
    unique_task_agents, task_agent_indices, counts = np.unique(
        task_agents, return_inverse=True, return_counts=True
    )
    random_nums = rng.random(len(indices))
    offsets = np.cumsum([0] + list(counts)[:-1])
    all_new_indices = [
        indices[task_agent_indices == j][
            (random_nums[offset:offset + count] * count).astype(np.int64)
        ]
        for j, (count, offset) in enumerate(zip(counts, offsets))
    ]
    return np.concatenate(all_new_indices)


def bootstrap_sample(
    data: pd.DataFrame,
    categories: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Perform hierarchical bootstrap sampling.

    Resamples at each category level with replacement within parent groups.
    Categories should be ordered from coarsest to finest
    (e.g., ["task_family", "task_id", "run_id"]).

    Args:
        data: DataFrame containing evaluation runs.
        categories: List of column names for hierarchical bootstrap levels.
        rng: NumPy random generator for reproducibility.

    Returns:
        Resampled DataFrame.
    """
    has_run_id = "run_id" in categories
    categories = [c for c in categories if c != "run_id"]

    # Validate hierarchy: each child must be a subcategory of parent
    for i in range(len(categories) - 1):
        parent, child = categories[i], categories[i + 1]
        assert (
            data.groupby(child)[parent].nunique().max() == 1
        ), f"{child} is not a subcategory of {parent}"

    category_arrays = {cat: data[cat].to_numpy() for cat in categories}

    indices = np.arange(len(data))
    split_ids = np.zeros(len(data), dtype=np.int32)
    new_split_id = 0

    for i, category in enumerate(categories):
        is_last = i == len(categories) - 1
        all_new_indices = []
        all_new_split_ids = []

        for group_id in np.unique(split_ids):
            group_indices = indices[split_ids == group_id]
            category_values = category_arrays[category][group_indices]

            values, value_indices = np.unique(category_values, return_inverse=True)
            n_values = len(values)
            sampled_values = rng.choice(n_values, size=n_values, replace=True)

            for j, sampled_value in enumerate(sampled_values):
                sampled_indices = group_indices[value_indices == sampled_value]
                all_new_indices.append(sampled_indices)
                if not is_last:
                    all_new_split_ids.append(
                        np.full(len(sampled_indices), new_split_id + j)
                    )
            new_split_id += n_values

        indices = np.concatenate(all_new_indices)
        if not is_last:
            split_ids = np.concatenate(all_new_split_ids)

    if has_run_id:
        task_col = data["task_id"].to_numpy()
        agent_col = data["agent"].to_numpy()
        indices = bootstrap_runs_by_task_agent(task_col, agent_col, indices, rng)

    return data.iloc[indices].copy()


def _process_single_bootstrap(
    bootstrap_idx: int,
    data: pd.DataFrame,
    categories: list[str],
    weights_col: str,
    regularization: float,
    rng: np.random.Generator,
    success_percents: list[int],
    score_col: str,
) -> dict[str, float]:
    """Process a single bootstrap iteration."""
    bootstrap_data = bootstrap_sample(data, categories, rng)
    results: dict[str, float] = {}

    for agent_name in bootstrap_data["agent"].unique():
        agent_data = bootstrap_data[bootstrap_data["agent"] == agent_name]
        x = np.log2(agent_data["human_minutes"].values).reshape(-1, 1)
        y = agent_data[score_col].values
        weights = agent_data[weights_col].values

        if len(np.unique(y)) < 2:
            continue

        model = fit_logistic(
            x, y, sample_weight=weights,
            regularization=regularization,
            ensure_weights_sum_to_1=False,
        )

        for sp in success_percents:
            q = sp / 100.0
            log_odds = np.log(q / (1 - q))
            x_val = (log_odds - model.intercept_[0]) / model.coef_[0][0]
            horizon = float(np.exp2(x_val))
            if np.isnan(horizon):
                logger.warning(
                    f"{agent_name} has NaN p{sp} on bootstrap {bootstrap_idx}"
                )
            results[f"{agent_name}_p{sp}"] = horizon

    return results


def compute_bootstrap_horizons(
    data: pd.DataFrame,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    regularization: float = DEFAULT_REGULARIZATION,
    weights_col: str = "weight",
    success_percents: list[int] | None = None,
    score_col: str = DEFAULT_SCORE_COL,
    seed: int = DEFAULT_SEED,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Run bootstrapped logistic regressions to estimate horizon distributions.

    Uses hierarchical bootstrap over [task_family, task_id, run_id].
    Parallelized with joblib in batches of 10.

    Args:
        data: DataFrame with columns [agent, task_id, task_family, run_id,
              human_minutes, <score_col>, <weights_col>].
        n_bootstrap: Number of bootstrap iterations.
        regularization: Logistic regression regularization.
        weights_col: Column name for sample weights.
        success_percents: List of success percentiles.
        score_col: Column name for scores.
        seed: Base random seed (iteration i uses seed+i).
        n_jobs: Number of parallel jobs (-1 = all cores minus one).

    Returns:
        DataFrame with rows=iterations, columns="{agent}_p{percent}".
    """
    if success_percents is None:
        success_percents = DEFAULT_SUCCESS_PERCENTS

    categories = ["task_family", "task_id", "run_id"]
    n_jobs_effective = max(1, Parallel(n_jobs=n_jobs)._effective_n_jobs())
    batch_size = 10
    n_batches = (n_bootstrap + batch_size - 1) // batch_size

    def process_batch(batch_idx: int) -> list[dict[str, float]]:
        start = batch_idx * batch_size
        end = min(start + batch_size, n_bootstrap)
        batch_results = []
        for i in range(start, end):
            rng = np.random.default_rng(seed + i)
            result = _process_single_bootstrap(
                i, data, categories, weights_col,
                regularization, rng, success_percents, score_col,
            )
            batch_results.append(result)
        return batch_results

    batched_results = Parallel(n_jobs=n_jobs_effective, verbose=1)(
        delayed(process_batch)(i) for i in range(n_batches)
    )

    results = [r for batch in batched_results for r in batch]
    return pd.DataFrame(results)


# ── Section 7: SOTA Determination & Trend Analysis ──────────────────────────


def _date_to_numeric(date_str: str) -> float:
    """Convert date string to numeric value for regression.

    Uses matplotlib's date2num if available, otherwise days since epoch.
    """
    dt = pd.to_datetime(date_str)
    if HAS_MATPLOTLIB:
        return float(date2num(dt))
    # Fallback: days since Unix epoch
    epoch = pd.Timestamp("1970-01-01")
    return float((dt - epoch).days)


def determine_sota_agents(
    agent_results: dict[str, dict[str, Any]],
    release_dates: dict[str, str],
    after_date: str | None = None,
    before_date: str | None = None,
) -> list[str]:
    """Determine which agents are state-of-the-art based on p50 at release time.

    An agent is SOTA if its p50 >= the highest p50 among all agents released
    on or before the same date.

    Args:
        agent_results: Dict mapping agent name to result dict (must have 'p50').
        release_dates: Dict mapping agent name to release date string.
        after_date: Only return SOTA agents released on or after this date.
        before_date: Only return SOTA agents released before this date.

    Returns:
        List of SOTA agent names.
    """
    agents_with_dates = []
    for name, res in agent_results.items():
        if name not in release_dates:
            continue
        p50 = res.get("p50", 0)
        if pd.isna(p50) or np.isinf(p50):
            continue
        agents_with_dates.append({
            "agent": name,
            "release_date": pd.to_datetime(release_dates[name]).date(),
            "p50": p50,
        })

    if not agents_with_dates:
        return []

    df = pd.DataFrame(agents_with_dates).sort_values("release_date")

    if after_date:
        df = df[df["release_date"] >= pd.to_datetime(after_date).date()]
    if before_date:
        df = df[df["release_date"] < pd.to_datetime(before_date).date()]

    sota = []
    highest = float("-inf")
    for release_date in df["release_date"].unique():
        agents_on_date = df[df["release_date"] == release_date]
        max_on_date = agents_on_date["p50"].max()
        highest = max(highest, max_on_date)
        for _, row in agents_on_date.iterrows():
            if row["p50"] >= highest:
                sota.append(row["agent"])

    return sota


def compute_doubling_time(
    p50s: list[float],
    dates: list[str],
) -> tuple[float, float]:
    """Compute doubling time from p50 horizons and release dates.

    Fits: log(p50) = slope * date_numeric + intercept
    doubling_time = log(2) / slope

    Args:
        p50s: List of p50 horizon values (minutes).
        dates: List of date strings corresponding to p50s.

    Returns:
        Tuple of (doubling_time_in_days, r_squared).
    """
    X = np.array([_date_to_numeric(d) for d in dates]).reshape(-1, 1)
    y = np.log(np.clip(p50s, 1e-3, np.inf))

    reg = LinearRegression().fit(X, y)
    r_squared = float(reg.score(X, y))
    doubling_time = float(np.log(2) / reg.coef_[0])

    return doubling_time, r_squared


def compute_trend_with_ci(
    agent_results: dict[str, dict[str, Any]],
    bootstrap_horizons: pd.DataFrame,
    release_dates: dict[str, str],
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    after_date: str = "2019-01-01",
    before_date: str = "2030-01-01",
) -> dict[str, Any]:
    """Compute doubling time with bootstrap confidence intervals.

    For each bootstrap sample, fits a trendline through SOTA agents' p50s
    and extracts the doubling time. Returns percentile-based CIs.

    Args:
        agent_results: Dict of agent results (must have 'p50').
        bootstrap_horizons: Bootstrap results DataFrame.
        release_dates: Dict mapping agent name to date string.
        confidence_level: Confidence level (e.g., 0.95).
        after_date: Start date for trend analysis.
        before_date: End date for trend analysis.

    Returns:
        Dict with point_estimate, ci_low, ci_high for doubling time in days.
    """
    # Determine SOTA agents from point estimates
    sota_agents = determine_sota_agents(
        agent_results, release_dates, after_date, before_date
    )
    if len(sota_agents) < 2:
        logger.warning("Need at least 2 SOTA agents for trend analysis")
        return {"point_estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}

    # Point estimate
    sota_p50s = [agent_results[a]["p50"] for a in sota_agents]
    sota_dates = [release_dates[a] for a in sota_agents]
    point_estimate, _ = compute_doubling_time(sota_p50s, sota_dates)

    # Bootstrap CIs
    p50_cols = [c for c in bootstrap_horizons.columns if c.endswith("_p50")]
    bs_p50 = bootstrap_horizons[p50_cols].copy()
    bs_p50.columns = pd.Index([c.removesuffix("_p50") for c in bs_p50.columns])

    doubling_times = []
    for sample_idx in range(len(bs_p50)):
        p50s_dates = []
        for agent in sota_agents:
            if agent not in bs_p50.columns:
                continue
            p50 = bs_p50[agent].iloc[sample_idx]
            if pd.isna(p50) or np.isinf(p50) or p50 < 1e-3:
                continue
            p50s_dates.append((float(p50), release_dates[agent]))

        if len(p50s_dates) < 2:
            continue

        p50_vals = [x[0] for x in p50s_dates]
        date_vals = [x[1] for x in p50s_dates]
        dt, _ = compute_doubling_time(p50_vals, date_vals)
        if dt > 0:
            doubling_times.append(dt)

    low_q = (1 - confidence_level) / 2
    high_q = 1 - low_q

    if doubling_times:
        ci_low = float(np.quantile(doubling_times, low_q))
        ci_high = float(np.quantile(doubling_times, high_q))
    else:
        ci_low = float("nan")
        ci_high = float("nan")

    return {
        "point_estimate": round(point_estimate, 3),
        "ci_low": round(ci_low, 3),
        "ci_high": round(ci_high, 3),
    }


# ── Section 8: Output Formatting ────────────────────────────────────────────


def format_results(
    agent_results: dict[str, dict[str, Any]],
    release_dates: dict[str, str] | None = None,
    doubling_time_stats: dict[str, Any] | None = None,
    benchmark_name: str = "METR-Horizon",
    fmt: str = "yaml",
) -> str:
    """Format results for output.

    Args:
        agent_results: Dict mapping agent name to result dict.
        release_dates: Optional dict of release dates.
        doubling_time_stats: Optional doubling time statistics.
        benchmark_name: Benchmark name string.
        fmt: Output format ("yaml", "json", "csv").

    Returns:
        Formatted string.
    """
    if fmt == "csv":
        return _format_csv(agent_results)
    output = _build_output_dict(
        agent_results, release_dates, doubling_time_stats, benchmark_name
    )
    if fmt == "json":
        return json.dumps(output, indent=2, default=str)
    return yaml.dump(output, default_flow_style=False, sort_keys=True)


def _build_output_dict(
    agent_results: dict[str, dict[str, Any]],
    release_dates: dict[str, str] | None,
    doubling_time_stats: dict[str, Any] | None,
    benchmark_name: str,
) -> dict[str, Any]:
    """Build the output dictionary matching METR benchmark_results format."""
    results = {}

    # Determine SOTA status
    sota_set = set()
    if release_dates:
        sota_set = set(determine_sota_agents(agent_results, release_dates))

    for agent_name, res in agent_results.items():
        agent_key = agent_name.lower().replace(" ", "_").replace("-", "_")
        metrics: dict[str, Any] = {
            "average_score": {"estimate": round(res["average_score"], 6)},
            "is_sota": agent_name in sota_set,
        }
        for p in DEFAULT_SUCCESS_PERCENTS:
            key = f"p{p}"
            if key in res:
                metrics[f"p{p}_horizon_length"] = {
                    "estimate": round(res[key], 6),
                    "ci_low": round(res.get(f"{key}_ci_low", float("nan")), 6),
                    "ci_high": round(res.get(f"{key}_ci_high", float("nan")), 6),
                }

        entry: dict[str, Any] = {
            "benchmark_name": benchmark_name,
            "metrics": metrics,
        }
        if release_dates and agent_name in release_dates:
            entry["release_date"] = str(release_dates[agent_name])

        results[agent_key] = entry

    output: dict[str, Any] = {
        "benchmark_name": benchmark_name,
        "results": results,
    }
    if doubling_time_stats:
        output["doubling_time_in_days"] = doubling_time_stats
    return output


def _format_csv(agent_results: dict[str, dict[str, Any]]) -> str:
    """Format results as CSV."""
    rows = []
    for name, res in agent_results.items():
        row = {"agent": name, "average_score": res["average_score"]}
        for p in DEFAULT_SUCCESS_PERCENTS:
            key = f"p{p}"
            if key in res:
                row[f"p{p}_horizon"] = res[key]
                row[f"p{p}_ci_low"] = res.get(f"{key}_ci_low", "")
                row[f"p{p}_ci_high"] = res.get(f"{key}_ci_high", "")
        row["coefficient"] = res.get("coefficient", "")
        row["intercept"] = res.get("intercept", "")
        rows.append(row)
    return pd.DataFrame(rows).to_csv(index=False)


# ── Section 9: Optional Plotting ─────────────────────────────────────────────


def plot_horizons(
    agent_results: dict[str, dict[str, Any]],
    release_dates: dict[str, str],
    success_percent: int = 50,
    output_path: str | None = None,
    show_trendline: bool = True,
) -> None:
    """Plot time horizons over release dates with optional trendline.

    Args:
        agent_results: Dict of agent results.
        release_dates: Dict mapping agent name to release date string.
        success_percent: Which horizon to plot (50 or 80).
        output_path: Path to save the plot. If None, displays interactively.
        show_trendline: Whether to show the exponential trendline.
    """
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required for plotting. Install with: pip install matplotlib")
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(12, 7))
    key = f"p{success_percent}"

    agents = []
    dates = []
    horizons = []
    ci_lows = []
    ci_highs = []

    for name, res in agent_results.items():
        if name not in release_dates or key not in res:
            continue
        p = res[key]
        if pd.isna(p) or np.isinf(p) or p <= 0:
            continue
        agents.append(name)
        dates.append(pd.to_datetime(release_dates[name]))
        horizons.append(p)
        ci_lows.append(res.get(f"{key}_ci_low", p))
        ci_highs.append(res.get(f"{key}_ci_high", p))

    horizons_arr = np.array(horizons)
    yerr = np.array([
        horizons_arr - np.array(ci_lows),
        np.array(ci_highs) - horizons_arr,
    ])
    yerr = np.clip(yerr, 0, np.inf)

    ax.errorbar(dates, horizons, yerr=yerr, fmt="none", ecolor="gray",
                capsize=3, alpha=0.5, zorder=1)
    scatter = ax.scatter(dates, horizons, s=80, zorder=2, c="steelblue", edgecolors="white")

    for i, name in enumerate(agents):
        ax.annotate(name, (dates[i], horizons[i]), textcoords="offset points",
                    xytext=(5, 5), fontsize=7, alpha=0.7)

    # Trendline through SOTA agents
    if show_trendline:
        sota = set(determine_sota_agents(agent_results, release_dates))
        sota_mask = [a in sota for a in agents]
        if sum(sota_mask) >= 2:
            sota_dates = [d for d, m in zip(dates, sota_mask) if m]
            sota_horizons = [h for h, m in zip(horizons, sota_mask) if m]
            X_num = np.array([date2num(d) for d in sota_dates]).reshape(-1, 1)
            y_log = np.log(np.array(sota_horizons))
            reg = LinearRegression().fit(X_num, y_log)
            x_range = np.linspace(X_num.min() - 30, X_num.max() + 90, 100)
            y_pred = np.exp(reg.predict(x_range.reshape(-1, 1)))
            ax.plot([num2date(x) for x in x_range], y_pred,
                    color="orangered", linewidth=2, alpha=0.7, linestyle="--")
            dt = np.log(2) / reg.coef_[0]
            ax.annotate(f"Doubling time: {dt:.0f} days",
                        xy=(0.02, 0.95), xycoords="axes fraction",
                        fontsize=11, color="orangered")

    ax.set_yscale("log")
    ax.set_xlabel("Model Release Date", fontsize=12)
    ax.set_ylabel(f"Task time (human minutes) at {success_percent}% success", fontsize=12)
    ax.set_title(f"{success_percent}% Time Horizon by Model Release Date", fontsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved to {output_path}")
    else:
        plt.show()
    plt.close(fig)


# ── Section 10: CLI Interface ────────────────────────────────────────────────


def _parse_column_map(raw: str | None) -> dict[str, str] | None:
    """Parse 'col1=col2,col3=col4' into a dict."""
    if not raw:
        return None
    result = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if "=" not in pair:
            raise ValueError(f"Invalid column mapping: '{pair}'. Use format: source=target")
        k, v = pair.split("=", 1)
        result[k.strip()] = v.strip()
    return result


def run_compute(args: argparse.Namespace) -> None:
    """Execute the compute subcommand."""
    success_percents = [int(x) for x in args.success_percents.split(",")]
    column_map = _parse_column_map(getattr(args, "column_map", None))
    auto_family = getattr(args, "auto_family", False)

    logger.info(f"Loading data from {args.input}")
    df = load_run_data(args.input, score_col=args.score_col,
                       column_map=column_map, auto_family=auto_family)
    logger.info(f"Loaded {len(df)} runs for {df['agent'].nunique()} agents")

    # Add weights
    df = add_weight_column(df, scheme=args.weighting)

    # Bootstrap (if requested)
    bootstrap_horizons = None
    if args.n_bootstrap > 0:
        logger.info(f"Running {args.n_bootstrap} bootstrap iterations...")
        bootstrap_horizons = compute_bootstrap_horizons(
            data=df,
            n_bootstrap=args.n_bootstrap,
            regularization=args.regularization,
            weights_col="weight",
            success_percents=success_percents,
            score_col=args.score_col,
            seed=args.seed,
            n_jobs=args.n_jobs,
        )
        logger.info("Bootstrap complete")

    # Compute per-agent results
    agent_results: dict[str, dict[str, Any]] = {}
    for agent_name, agent_df in df.groupby("agent"):
        result = compute_agent_horizon(
            human_minutes=agent_df["human_minutes"].values,
            scores=agent_df[args.score_col].values,
            weights=agent_df["weight"].values,
            agent_name=str(agent_name),
            regularization=args.regularization,
            success_percents=success_percents,
            confidence_level=args.confidence_level,
            bootstrap_horizons=bootstrap_horizons,
        )
        agent_results[str(agent_name)] = result

    # Trend analysis
    doubling_time_stats = None
    release_dates = None
    if args.release_dates:
        release_dates = load_release_dates(args.release_dates)
        if not args.no_trend and bootstrap_horizons is not None:
            logger.info("Computing doubling time trend...")
            doubling_time_stats = {
                "from_2023_on": compute_trend_with_ci(
                    agent_results, bootstrap_horizons, release_dates,
                    confidence_level=args.confidence_level,
                    after_date="2023-01-01", before_date="2030-01-01",
                ),
            }

    # Output
    output_str = format_results(
        agent_results,
        release_dates=release_dates,
        doubling_time_stats=doubling_time_stats,
        benchmark_name=args.benchmark_name,
        fmt=args.format,
    )
    if args.output:
        pathlib.Path(args.output).write_text(output_str)
        logger.info(f"Results written to {args.output}")
    else:
        print(output_str)


def run_validate(args: argparse.Namespace) -> None:
    """Execute the validate subcommand.

    Matches agents to model keys using:
      1. The 'model' column in the input data (exact match to benchmark keys)
      2. Fallback: normalized agent name matching
    """
    expected = yaml.safe_load(pathlib.Path(args.expected).read_text())
    expected_results = expected.get("results", {})

    # Load raw data to get model column if available
    path = pathlib.Path(args.input)
    if path.suffix == ".jsonl":
        df_raw = pd.read_json(path, lines=True, orient="records", convert_dates=False)
    else:
        df_raw = pd.read_csv(path)

    # Build alias-to-model mapping if 'model' column exists
    alias_to_model: dict[str, str] = {}
    if "model" in df_raw.columns:
        alias_col = "alias" if "alias" in df_raw.columns else "agent"
        for alias, group in df_raw.groupby(alias_col):
            models = group["model"].unique()
            if len(models) == 1:
                alias_to_model[str(alias)] = str(models[0])

    df = load_run_data(args.input, score_col=args.score_col)

    # Use precomputed weights if available, otherwise compute
    if args.weighting in df.columns:
        df["weight"] = df[args.weighting]
        # Verify normalization per agent
        for agent, agent_df in df.groupby("agent"):
            if not np.allclose(agent_df["weight"].sum(), 1.0, atol=1e-6):
                logger.info(f"Re-normalizing weights for {agent}")
                df.loc[agent_df.index, "weight"] = (
                    agent_df["weight"] / agent_df["weight"].sum()
                )
    else:
        df = add_weight_column(df, scheme=args.weighting)

    agent_results: dict[str, dict[str, Any]] = {}
    for agent_name, agent_df in df.groupby("agent"):
        result = compute_agent_horizon(
            human_minutes=agent_df["human_minutes"].values,
            scores=agent_df[args.score_col].values,
            weights=agent_df["weight"].values,
            agent_name=str(agent_name),
            regularization=args.regularization,
            success_percents=[50, 80],
            confidence_level=0.95,
        )
        agent_results[str(agent_name)] = result

    # Build model_key -> agent_name mapping
    model_key_to_agent: dict[str, str] = {}
    for agent_name in agent_results:
        model_key = alias_to_model.get(agent_name)
        if model_key:
            model_key_to_agent[model_key] = agent_name

    n_compared = 0
    n_passed = 0
    n_failed = 0
    tol = args.tolerance

    for model_key, expected_data in sorted(expected_results.items()):
        exp_metrics = expected_data.get("metrics", {})

        # Find matching agent
        agent_name = model_key_to_agent.get(model_key)
        if agent_name is None:
            # Fallback: try normalized name matching
            for a in agent_results:
                a_norm = a.lower().replace(" ", "_").replace("-", "_").replace(".", "_")
                if a_norm == model_key:
                    agent_name = a
                    break

        if agent_name is None:
            # Model might be from a stitched older benchmark version
            bm = expected_data.get("benchmark_name", "")
            if bm != expected.get("benchmark_name", ""):
                logger.info(f"  Skipping {model_key} (from {bm}, stitched from older benchmark)")
            else:
                logger.warning(f"  No match found for {model_key}, skipping")
            continue

        res = agent_results[agent_name]
        checks = [
            ("average_score", exp_metrics.get("average_score", {}).get("estimate"), res.get("average_score")),
            ("p50", exp_metrics.get("p50_horizon_length", {}).get("estimate"), res.get("p50")),
            ("p80", exp_metrics.get("p80_horizon_length", {}).get("estimate"), res.get("p80")),
        ]

        for metric_name, expected_val, actual_val in checks:
            if expected_val is None or actual_val is None:
                continue
            n_compared += 1
            if expected_val == 0:
                if abs(actual_val) < tol:
                    n_passed += 1
                else:
                    n_failed += 1
                    print(f"  FAIL {model_key}.{metric_name}: expected {expected_val}, got {actual_val}")
            else:
                rel_err = abs(actual_val - expected_val) / abs(expected_val)
                if rel_err <= tol:
                    n_passed += 1
                else:
                    n_failed += 1
                    print(f"  FAIL {model_key}.{metric_name}: expected {expected_val}, got {actual_val} (rel_err={rel_err:.4f})")

    print(f"\nValidation: {n_passed}/{n_compared} checks passed, {n_failed} failed (tolerance={tol})")
    if n_failed > 0:
        sys.exit(1)


def run_plot(args: argparse.Namespace) -> None:
    """Execute the plot subcommand."""
    data = yaml.safe_load(pathlib.Path(args.input).read_text())
    release_dates = load_release_dates(args.release_dates)

    # Parse results back to agent_results format
    agent_results: dict[str, dict[str, Any]] = {}
    for model_key, model_data in data.get("results", {}).items():
        metrics = model_data.get("metrics", {})
        res: dict[str, Any] = {
            "average_score": metrics.get("average_score", {}).get("estimate", 0),
        }
        for p in [50, 80]:
            h = metrics.get(f"p{p}_horizon_length", {})
            if h:
                res[f"p{p}"] = h.get("estimate", 0)
                res[f"p{p}_ci_low"] = h.get("ci_low", 0)
                res[f"p{p}_ci_high"] = h.get("ci_high", 0)
        # Try to find agent name in release dates
        rd = model_data.get("release_date")
        agent_name = model_key
        for name, date in release_dates.items():
            if str(date) == str(rd):
                nk = name.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
                if model_key.startswith(nk[:8]) or nk.startswith(model_key[:8]):
                    agent_name = name
                    break
        agent_results[agent_name] = res

    plot_horizons(
        agent_results, release_dates,
        success_percent=args.success_percent,
        output_path=args.output,
    )


def run_convert(args: argparse.Namespace) -> None:
    """Execute the convert subcommand.

    Converts arbitrary data formats into the standard input format.
    """
    column_map = _parse_column_map(getattr(args, "column_map", None))
    auto_family = getattr(args, "auto_family", False)

    df = load_run_data(args.input, score_col=args.score_col,
                       column_map=column_map, auto_family=auto_family)

    if args.preview:
        # Find the actual score column (may have been alias-resolved)
        score_col_actual = args.score_col if args.score_col in df.columns else "score_binarized"
        show_cols = ["task_id", "task_family", "agent", "human_minutes", score_col_actual, "run_id"]
        show_cols = [c for c in show_cols if c in df.columns]
        print("Converted data (first 5 rows):\n")
        print(df[show_cols].head().to_string(index=False))
        print(f"\n{len(df)} total rows, {df['agent'].nunique()} agents, "
              f"{df['task_id'].nunique()} tasks, {df['task_family'].nunique()} families")
        return

    out_path = pathlib.Path(args.output)
    if out_path.suffix == ".jsonl":
        df.to_json(out_path, orient="records", lines=True)
    elif out_path.suffix == ".csv":
        df.to_csv(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
    print(f"Converted {len(df)} rows -> {args.output}")


def run_selftest(_args: argparse.Namespace) -> None:
    """Execute built-in self-tests."""
    print("Running self-tests...\n")
    passed = 0
    failed = 0

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f"  PASS: {name}")
        else:
            failed += 1
            print(f"  FAIL: {name} {detail}")

    # Test 1: Weight computation
    print("Test 1: Weight computation")
    df_test = pd.DataFrame({
        "task_id": ["t1", "t1", "t2", "t3", "t3", "t3"],
        "task_family": ["f1", "f1", "f1", "f2", "f2", "f2"],
        "agent": ["A"] * 6,
        "run_id": [f"r{i}" for i in range(6)],
        "human_minutes": [10, 10, 20, 30, 30, 30],
        "score_binarized": [1, 0, 1, 0, 0, 1],
    })
    w_equal = compute_task_weights(df_test, "equal_task_weight")
    check("equal weights sum to 1", np.allclose(w_equal.sum(), 1.0))
    # t1 has 2 runs, t2 has 1, t3 has 3. Each task gets equal total weight.
    t1_total = w_equal.iloc[:2].sum()
    t2_total = w_equal.iloc[2:3].sum()
    t3_total = w_equal.iloc[3:].sum()
    check("equal weights per task", np.allclose(t1_total, t2_total) and np.allclose(t2_total, t3_total),
          f"t1={t1_total:.4f} t2={t2_total:.4f} t3={t3_total:.4f}")

    w_invsqrt = compute_task_weights(df_test, "invsqrt_task_weight")
    check("invsqrt weights sum to 1", np.allclose(w_invsqrt.sum(), 1.0))

    # Test 2: Logistic fit on known data
    print("\nTest 2: Logistic regression on known data")
    np.random.seed(42)
    # Generate data from logistic: P(success) = sigmoid(-2 * log2(x) + 6)
    # p50 at: 0 = -2*log2(x) + 6 => log2(x) = 3 => x = 8 minutes
    n = 500
    x_mins = np.exp2(np.random.uniform(0, 6, n))  # 1 to 64 minutes
    true_coef, true_intercept = -2.0, 6.0
    probs = 1.0 / (1.0 + np.exp(-(true_coef * np.log2(x_mins) + true_intercept)))
    y = (np.random.rand(n) < probs).astype(float)
    weights = np.ones(n) / n
    X = np.log2(x_mins).reshape(-1, 1)
    model = fit_logistic(X, y, weights, regularization=1e-5)
    check("coefficient is negative", model.coef_[0][0] < 0)
    fitted_p50 = get_horizon_at_percent(model, 50)
    check("p50 near 8 minutes", abs(fitted_p50 - 8.0) < 3.0,
          f"got {fitted_p50:.2f}")

    # Test 3: Fractional y splitting
    print("\nTest 3: Fractional y splitting")
    X_frac = np.array([[1], [2], [3]], dtype=float)
    y_frac = np.array([0.3, 0.7, 0.5])
    w_frac = np.array([0.33, 0.34, 0.33])
    orig_avg = np.average(y_frac, weights=w_frac)
    model_frac = fit_logistic(X_frac, y_frac, w_frac, ensure_weights_sum_to_1=False)
    check("fractional fit succeeds", model_frac is not None)

    # Test 4: Horizon extraction
    print("\nTest 4: Horizon extraction formula")
    # Create a model with known parameters
    mock_model = LogisticRegression()
    mock_model.coef_ = np.array([[-1.0]])
    mock_model.intercept_ = np.array([3.0])
    mock_model.classes_ = np.array([0, 1])
    # p50: log(1) = 0 => x = (0 - 3) / (-1) = 3 => 2^3 = 8
    p50 = get_horizon_at_percent(mock_model, 50)
    check("p50 formula correct", np.allclose(p50, 8.0), f"got {p50}")
    # p80: log(4) ≈ 1.386 => x = (1.386 - 3) / (-1) = 1.614 => 2^1.614 ≈ 3.06
    p80 = get_horizon_at_percent(mock_model, 80)
    expected_p80 = 2 ** ((np.log(0.8 / 0.2) - 3.0) / (-1.0))
    check("p80 formula correct", np.allclose(p80, expected_p80), f"got {p80}, expected {expected_p80}")

    # Test 5: SOTA determination
    print("\nTest 5: SOTA determination")
    test_results = {
        "A": {"p50": 10.0},
        "B": {"p50": 5.0},
        "C": {"p50": 20.0},
        "D": {"p50": 15.0},
    }
    test_dates = {
        "A": "2024-01-01",
        "B": "2024-02-01",
        "C": "2024-03-01",
        "D": "2024-04-01",
    }
    sota = determine_sota_agents(test_results, test_dates)
    check("SOTA includes A", "A" in sota)
    check("SOTA excludes B", "B" not in sota, f"got {sota}")
    check("SOTA includes C", "C" in sota)
    check("SOTA excludes D", "D" not in sota, f"got {sota}")

    # Test 6: Doubling time
    print("\nTest 6: Doubling time computation")
    # Create points that double every 100 days
    test_p50s = [1.0, 2.0, 4.0, 8.0]
    base = datetime.date(2024, 1, 1)
    test_dates_dt = [str(base + datetime.timedelta(days=i * 100)) for i in range(4)]
    dt, r2 = compute_doubling_time(test_p50s, test_dates_dt)
    check("doubling time near 100 days", abs(dt - 100) < 5, f"got {dt:.1f}")
    check("R^2 near 1.0", r2 > 0.99, f"got {r2:.4f}")

    # Test 7: Bootstrap smoke test
    print("\nTest 7: Bootstrap smoke test")
    df_bs = pd.DataFrame({
        "task_id": (["t1"] * 4 + ["t2"] * 4) * 2,
        "task_family": (["f1"] * 4 + ["f2"] * 4) * 2,
        "agent": ["A"] * 8 + ["B"] * 8,
        "run_id": [f"r{i}" for i in range(16)],
        "human_minutes": [5, 10, 20, 40, 5, 10, 20, 40] * 2,
        "score_binarized": [1, 1, 0, 0, 1, 0, 0, 0] * 2,
    })
    df_bs = add_weight_column(df_bs, "equal_task_weight")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bs_results = compute_bootstrap_horizons(
            df_bs, n_bootstrap=10, weights_col="weight",
            success_percents=[50], seed=42, n_jobs=1,
        )
    check("bootstrap returns DataFrame", isinstance(bs_results, pd.DataFrame))
    check("bootstrap has correct rows", len(bs_results) == 10, f"got {len(bs_results)}")

    # Test 8: Full pipeline integration
    print("\nTest 8: Full pipeline integration")
    df_full = pd.DataFrame({
        "task_id": ["t1"] * 6 + ["t2"] * 6 + ["t3"] * 6,
        "task_family": ["f1"] * 12 + ["f2"] * 6,
        "agent": (["X", "Y"] * 3) * 3,
        "run_id": [f"r{i}" for i in range(18)],
        "human_minutes": [5, 5, 20, 20, 60, 60] * 3,
        "score_binarized": [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
    })
    df_full = add_weight_column(df_full, "invsqrt_task_weight")
    res_x = compute_agent_horizon(
        human_minutes=df_full[df_full["agent"] == "X"]["human_minutes"].values,
        scores=df_full[df_full["agent"] == "X"]["score_binarized"].values,
        weights=df_full[df_full["agent"] == "X"]["weight"].values,
        agent_name="X", success_percents=[50, 80],
    )
    check("agent result has p50", "p50" in res_x)
    check("agent result has p80", "p80" in res_x)
    check("p50 > 0", res_x["p50"] > 0, f"got {res_x['p50']}")

    # Test 9: Output formatting
    print("\nTest 9: Output formatting")
    test_agent_results = {
        "TestAgent": {
            "average_score": 0.5, "coefficient": -1.0, "intercept": 3.0,
            "p50": 8.0, "p50_ci_low": 4.0, "p50_ci_high": 16.0,
            "p80": 3.0, "p80_ci_low": 1.0, "p80_ci_high": 6.0,
        }
    }
    yaml_out = format_results(test_agent_results, fmt="yaml")
    check("YAML output parseable", yaml.safe_load(yaml_out) is not None)
    json_out = format_results(test_agent_results, fmt="json")
    check("JSON output parseable", json.loads(json_out) is not None)
    csv_out = format_results(test_agent_results, fmt="csv")
    check("CSV output non-empty", len(csv_out) > 0)

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
    print("All self-tests passed!")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="Standalone METR Time Horizon computation script.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # compute
    p_compute = subparsers.add_parser("compute", help="Compute time horizons from run data")
    p_compute.add_argument("--input", required=True, help="Path to run data (JSONL or CSV)")
    p_compute.add_argument("--release-dates", help="Path to release dates YAML")
    p_compute.add_argument("--output", help="Output file path (default: stdout)")
    p_compute.add_argument("--format", choices=["yaml", "json", "csv"], default="yaml")
    p_compute.add_argument("--weighting", default=DEFAULT_WEIGHTING,
                           choices=["equal_task_weight", "invsqrt_task_weight"])
    p_compute.add_argument("--regularization", type=float, default=DEFAULT_REGULARIZATION)
    p_compute.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    p_compute.add_argument("--success-percents", default="50,80",
                           help="Comma-separated success percentiles")
    p_compute.add_argument("--confidence-level", type=float, default=DEFAULT_CONFIDENCE_LEVEL)
    p_compute.add_argument("--score-col", default=DEFAULT_SCORE_COL)
    p_compute.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_compute.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs (-1=auto)")
    p_compute.add_argument("--no-trend", action="store_true", help="Skip trend analysis")
    p_compute.add_argument("--benchmark-name", default="METR-Horizon")
    p_compute.add_argument("--column-map",
                           help="Column mapping as key=value pairs, e.g.: model=agent,duration_hrs=human_hours")
    p_compute.add_argument("--auto-family", action="store_true",
                           help="Auto-generate task_family from task_id prefixes if missing")

    # convert
    p_convert = subparsers.add_parser(
        "convert",
        help="Convert external data formats to the standard input format",
        description="Converts arbitrary CSV/JSONL data into the required format. "
                    "Use --column-map to rename columns and --auto-family if task_family is missing.",
    )
    p_convert.add_argument("--input", required=True, help="Input data file")
    p_convert.add_argument("--output", required=True, help="Output file path (.csv or .jsonl)")
    p_convert.add_argument("--column-map",
                           help="Column mapping as key=value pairs, e.g.: model=agent,hours=human_hours")
    p_convert.add_argument("--auto-family", action="store_true",
                           help="Auto-generate task_family from task_id prefixes")
    p_convert.add_argument("--score-col", default=DEFAULT_SCORE_COL,
                           help="Score column name in your data")
    p_convert.add_argument("--preview", action="store_true",
                           help="Show first 5 rows of converted data instead of writing")

    # validate
    p_validate = subparsers.add_parser("validate", help="Validate against known results")
    p_validate.add_argument("--input", required=True, help="Path to run data")
    p_validate.add_argument("--expected", required=True, help="Path to expected results YAML")
    p_validate.add_argument("--release-dates", help="Path to release dates YAML")
    p_validate.add_argument("--tolerance", type=float, default=0.01)
    p_validate.add_argument("--weighting", default=DEFAULT_WEIGHTING)
    p_validate.add_argument("--regularization", type=float, default=DEFAULT_REGULARIZATION)
    p_validate.add_argument("--score-col", default=DEFAULT_SCORE_COL)

    # selftest
    subparsers.add_parser("selftest", help="Run built-in self-tests")

    # plot
    p_plot = subparsers.add_parser("plot", help="Plot time horizons")
    p_plot.add_argument("--input", required=True, help="Path to computed results YAML")
    p_plot.add_argument("--release-dates", required=True, help="Path to release dates YAML")
    p_plot.add_argument("--output", help="Output image path")
    p_plot.add_argument("--success-percent", type=int, default=50)

    return parser


def main() -> None:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO if args.verbose else logging.WARNING,
    )

    if args.command == "compute":
        run_compute(args)
    elif args.command == "convert":
        run_convert(args)
    elif args.command == "validate":
        run_validate(args)
    elif args.command == "selftest":
        run_selftest(args)
    elif args.command == "plot":
        run_plot(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
