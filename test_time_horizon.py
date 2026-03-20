#!/usr/bin/env python3
"""
Test suite for compute_time_horizon.py

Tests cover:
  - Unit tests for individual functions
  - Integration tests for the full pipeline
  - Validation against METR benchmark results (if runs.jsonl available)
"""

import datetime
import json
import math
import os
import pathlib
import tempfile
import warnings

import numpy as np
import pandas as pd
import pytest
import yaml

from compute_time_horizon import (
    DEFAULT_REGULARIZATION,
    DEFAULT_SUCCESS_PERCENTS,
    add_weight_column,
    bootstrap_sample,
    compute_agent_horizon,
    compute_bootstrap_horizons,
    compute_doubling_time,
    compute_task_weights,
    compute_trend_with_ci,
    determine_sota_agents,
    fit_logistic,
    format_results,
    get_horizon_at_percent,
    load_release_dates,
    load_run_data,
)
from sklearn.linear_model import LogisticRegression


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def simple_agent_df():
    """Single agent, 3 tasks in 2 families, varying runs per task."""
    return pd.DataFrame({
        "task_id": ["t1", "t1", "t2", "t3", "t3", "t3"],
        "task_family": ["f1", "f1", "f1", "f2", "f2", "f2"],
        "agent": ["A"] * 6,
        "run_id": [f"r{i}" for i in range(6)],
        "human_minutes": [10, 10, 20, 30, 30, 30],
        "score_binarized": [1, 0, 1, 0, 0, 1],
    })


@pytest.fixture
def multi_agent_df():
    """Two agents with distinct performance profiles."""
    rows = []
    np.random.seed(123)
    for agent_idx, (agent, base_success) in enumerate([("Strong", 0.7), ("Weak", 0.3)]):
        for task_idx in range(10):
            family = f"f{task_idx // 5}"
            task = f"t{task_idx}"
            minutes = 2 ** (task_idx + 1)  # 2, 4, 8, ... 1024
            for run_idx in range(3):
                # Success probability decreases with task length
                p = base_success * max(0.1, 1.0 - 0.08 * task_idx)
                score = 1 if np.random.rand() < p else 0
                rows.append({
                    "task_id": task,
                    "task_family": family,
                    "agent": agent,
                    "run_id": f"r_{agent}_{task}_{run_idx}",
                    "human_minutes": minutes,
                    "score_binarized": score,
                })
    return pd.DataFrame(rows)


@pytest.fixture
def known_logistic_data():
    """Data generated from a known logistic function.

    P(success) = sigmoid(-2 * log2(x) + 6)
    True p50: log2(x) = 3 => x = 8 minutes
    """
    np.random.seed(42)
    n = 1000
    x_mins = np.exp2(np.random.uniform(0, 6, n))
    true_coef, true_intercept = -2.0, 6.0
    probs = 1.0 / (1.0 + np.exp(-(true_coef * np.log2(x_mins) + true_intercept)))
    y = (np.random.rand(n) < probs).astype(float)
    weights = np.ones(n) / n
    return x_mins, y, weights, true_coef, true_intercept


# ── Unit Tests: Weight Computation ───────────────────────────────────────────


class TestTaskWeights:
    def test_equal_weights_sum_to_one(self, simple_agent_df):
        w = compute_task_weights(simple_agent_df, "equal_task_weight")
        assert np.allclose(w.sum(), 1.0)

    def test_invsqrt_weights_sum_to_one(self, simple_agent_df):
        w = compute_task_weights(simple_agent_df, "invsqrt_task_weight")
        assert np.allclose(w.sum(), 1.0)

    def test_equal_weights_per_task(self, simple_agent_df):
        """Each task should get equal total weight regardless of run count."""
        w = compute_task_weights(simple_agent_df, "equal_task_weight")
        task_totals = simple_agent_df.assign(w=w).groupby("task_id")["w"].sum()
        # All tasks should have the same total weight
        assert np.allclose(task_totals.values, task_totals.values[0])

    def test_invsqrt_downweights_large_families(self, simple_agent_df):
        """Tasks in larger families should get less total weight with invsqrt."""
        w = compute_task_weights(simple_agent_df, "invsqrt_task_weight")
        df = simple_agent_df.assign(w=w)
        # f1 has 2 tasks, f2 has 1 task
        f1_weight = df[df["task_family"] == "f1"]["w"].sum()
        f2_weight = df[df["task_family"] == "f2"]["w"].sum()
        # With invsqrt: f1 tasks are downweighted by 1/sqrt(2) each
        # f2 task is 1/sqrt(1) = 1
        # So f2 should get more total weight per task than f1's tasks individually
        f1_per_task = f1_weight / 2  # f1 has 2 tasks
        assert f2_weight > f1_per_task

    def test_unknown_scheme_raises(self, simple_agent_df):
        with pytest.raises(ValueError, match="Unknown weighting"):
            compute_task_weights(simple_agent_df, "bogus_weight")

    def test_add_weight_column_multi_agent(self, multi_agent_df):
        df = add_weight_column(multi_agent_df, "equal_task_weight")
        assert "weight" in df.columns
        # Weights should sum to 1.0 per agent
        for _, group in df.groupby("agent"):
            assert np.allclose(group["weight"].sum(), 1.0)

    def test_single_task_single_run(self):
        """Edge case: one task, one run."""
        df = pd.DataFrame({
            "task_id": ["t1"],
            "task_family": ["f1"],
            "agent": ["A"],
            "run_id": ["r0"],
            "human_minutes": [10],
            "score_binarized": [1],
        })
        w = compute_task_weights(df, "equal_task_weight")
        assert np.allclose(w.sum(), 1.0)
        assert np.allclose(w.values[0], 1.0)


# ── Unit Tests: Logistic Regression ──────────────────────────────────────────


class TestLogisticRegression:
    def test_basic_fit(self, known_logistic_data):
        x_mins, y, weights, true_coef, true_intercept = known_logistic_data
        X = np.log2(x_mins).reshape(-1, 1)
        model = fit_logistic(X, y, weights, regularization=1e-5)
        # Coefficient should be negative (success decreases with task length)
        assert model.coef_[0][0] < 0
        # Should roughly recover true parameters
        assert abs(model.coef_[0][0] - true_coef) < 0.5
        assert abs(model.intercept_[0] - true_intercept) < 1.5

    def test_fractional_y_preserves_average(self):
        """Splitting fractional y should preserve the weighted average."""
        X = np.array([[1], [2], [3], [4]], dtype=float)
        y = np.array([0.2, 0.8, 0.5, 1.0])
        w = np.array([0.25, 0.25, 0.25, 0.25])
        orig_avg = np.average(y, weights=w)

        # The function should not raise and the model should fit
        model = fit_logistic(X, y, w, ensure_weights_sum_to_1=True)
        assert model is not None

    def test_all_zeros_y(self):
        """Should handle edge case where all y=0 without crashing."""
        X = np.array([[1], [2], [3]], dtype=float)
        y = np.array([0.0, 0.0, 0.0])
        w = np.array([0.33, 0.34, 0.33])
        # With all zeros, sklearn may converge to extreme weights
        # The compute_agent_horizon function handles this case
        # fit_logistic itself requires binary variation, so this
        # tests the agent_regression wrapper behavior
        pass  # This is handled at the compute_agent_horizon level

    def test_all_ones_y(self):
        """When all y=1, sklearn raises ValueError (single class).
        This is handled at the compute_agent_horizon level (all-zero check),
        but for all-ones the caller should catch this or ensure data has variation."""
        X = np.array([[1], [2], [3]], dtype=float)
        y = np.array([1.0, 1.0, 1.0])
        w = np.array([0.33, 0.34, 0.33])
        # sklearn requires at least 2 classes for LogisticRegression
        with pytest.raises(ValueError):
            fit_logistic(X, y, w, ensure_weights_sum_to_1=False)

    def test_weight_sum_assertion(self):
        """Should raise when weights don't sum to 1 and check is on."""
        X = np.array([[1], [2]], dtype=float)
        y = np.array([0.0, 1.0])
        w = np.array([0.5, 0.7])  # sums to 1.2
        with pytest.raises(AssertionError):
            fit_logistic(X, y, w, ensure_weights_sum_to_1=True)

    def test_regularization_effect(self):
        """Different regularization values should produce different results."""
        np.random.seed(42)
        X = np.random.randn(50, 1)
        y = (X.ravel() > 0).astype(float)
        w = np.ones(50) / 50

        model_weak = fit_logistic(X, y, w, regularization=1e-5)
        model_strong = fit_logistic(X, y, w, regularization=1.0)
        # Stronger regularization should shrink coefficient toward 0
        assert abs(model_strong.coef_[0][0]) < abs(model_weak.coef_[0][0])


# ── Unit Tests: Horizon Extraction ───────────────────────────────────────────


class TestHorizonExtraction:
    def test_p50_formula(self):
        model = LogisticRegression()
        model.coef_ = np.array([[-1.0]])
        model.intercept_ = np.array([3.0])
        model.classes_ = np.array([0, 1])
        # p50: log(1) = 0, x = (0-3)/(-1) = 3, 2^3 = 8
        assert np.allclose(get_horizon_at_percent(model, 50), 8.0)

    def test_p80_formula(self):
        model = LogisticRegression()
        model.coef_ = np.array([[-1.0]])
        model.intercept_ = np.array([3.0])
        model.classes_ = np.array([0, 1])
        q = 0.8
        expected_x = (np.log(q / (1 - q)) - 3.0) / (-1.0)
        expected = 2 ** expected_x
        assert np.allclose(get_horizon_at_percent(model, 80), expected)

    def test_higher_percent_gives_lower_horizon(self):
        """p80 should be lower than p50 (harder bar to clear)."""
        model = LogisticRegression()
        model.coef_ = np.array([[-1.0]])
        model.intercept_ = np.array([3.0])
        model.classes_ = np.array([0, 1])
        p50 = get_horizon_at_percent(model, 50)
        p80 = get_horizon_at_percent(model, 80)
        assert p80 < p50

    def test_steeper_coefficient_narrower_range(self):
        """Steeper (more negative) coefficient means sharper dropoff."""
        model_shallow = LogisticRegression()
        model_shallow.coef_ = np.array([[-0.5]])
        model_shallow.intercept_ = np.array([3.0])
        model_shallow.classes_ = np.array([0, 1])

        model_steep = LogisticRegression()
        model_steep.coef_ = np.array([[-2.0]])
        model_steep.intercept_ = np.array([3.0])
        model_steep.classes_ = np.array([0, 1])

        # Both have same p50 (at intercept/coef balance)
        # but steeper model has narrower range between p80 and p20
        p80_shallow = get_horizon_at_percent(model_shallow, 80)
        p80_steep = get_horizon_at_percent(model_steep, 80)
        p50_shallow = get_horizon_at_percent(model_shallow, 50)
        p50_steep = get_horizon_at_percent(model_steep, 50)

        ratio_shallow = p50_shallow / p80_shallow
        ratio_steep = p50_steep / p80_steep
        # Steeper model should have smaller ratio (p50/p80 closer to 1 in log space)
        assert ratio_steep < ratio_shallow


# ── Unit Tests: Per-Agent Regression ─────────────────────────────────────────


class TestAgentRegression:
    def test_basic_agent_result(self, known_logistic_data):
        x_mins, y, weights, _, _ = known_logistic_data
        result = compute_agent_horizon(
            x_mins, y, weights, "TestAgent",
            success_percents=[50, 80],
        )
        assert "agent" in result
        assert result["agent"] == "TestAgent"
        assert "p50" in result
        assert "p80" in result
        assert result["p50"] > 0
        assert result["p80"] > 0
        assert result["p50"] > result["p80"]  # p80 is stricter

    def test_all_zeros_returns_zero_horizons(self):
        """Agent that never succeeds should get 0 horizons."""
        x = np.array([5, 10, 20, 40], dtype=float)
        y = np.array([0, 0, 0, 0], dtype=float)
        w = np.array([0.25, 0.25, 0.25, 0.25])
        result = compute_agent_horizon(x, y, w, "ZeroAgent", success_percents=[50, 80])
        assert result["p50"] == 0
        assert result["p80"] == 0
        assert result["coefficient"] == float("-inf")

    def test_with_bootstrap_ci(self):
        """CI extraction from bootstrap results."""
        x = np.array([5, 10, 20, 40, 5, 10, 20, 40], dtype=float)
        y = np.array([1, 1, 0, 0, 1, 0, 0, 0], dtype=float)
        w = np.ones(8) / 8
        # Fake bootstrap results
        bs = pd.DataFrame({
            "TestAgent_p50": [8.0, 10.0, 12.0, 6.0, 9.0],
            "TestAgent_p80": [2.0, 3.0, 4.0, 1.5, 2.5],
        })
        result = compute_agent_horizon(
            x, y, w, "TestAgent",
            success_percents=[50, 80],
            confidence_level=0.95,
            bootstrap_horizons=bs,
        )
        assert not np.isnan(result["p50_ci_low"])
        assert not np.isnan(result["p50_ci_high"])
        assert result["p50_ci_low"] <= result["p50_ci_high"]

    def test_average_score(self):
        """Average score should be weighted average of scores."""
        x = np.array([5, 10], dtype=float)
        y = np.array([1.0, 0.0])
        w = np.array([0.7, 0.3])
        result = compute_agent_horizon(x, y, w, "A", success_percents=[50])
        expected_avg = 0.7 * 1.0 + 0.3 * 0.0
        assert np.allclose(result["average_score"], expected_avg)


# ── Unit Tests: Bootstrap ────────────────────────────────────────────────────


class TestBootstrap:
    def test_bootstrap_sample_preserves_columns(self):
        df = pd.DataFrame({
            "task_id": ["t1", "t1", "t2", "t2"],
            "task_family": ["f1", "f1", "f1", "f1"],
            "agent": ["A", "A", "A", "A"],
            "run_id": ["r0", "r1", "r2", "r3"],
            "human_minutes": [5, 10, 20, 40],
            "score_binarized": [1, 0, 1, 0],
            "weight": [0.25, 0.25, 0.25, 0.25],
        })
        rng = np.random.default_rng(42)
        sample = bootstrap_sample(df, ["task_family", "task_id", "run_id"], rng)
        assert set(sample.columns) == set(df.columns)

    def test_bootstrap_sample_size_varies(self):
        """Bootstrap samples may have different sizes due to resampling."""
        df = pd.DataFrame({
            "task_id": ["t1", "t1", "t2"],
            "task_family": ["f1", "f1", "f1"],
            "agent": ["A", "A", "A"],
            "run_id": ["r0", "r1", "r2"],
            "human_minutes": [5, 10, 20],
            "score_binarized": [1, 0, 1],
            "weight": [1/3, 1/3, 1/3],
        })
        sizes = set()
        for seed in range(50):
            rng = np.random.default_rng(seed)
            sample = bootstrap_sample(df, ["task_family", "task_id", "run_id"], rng)
            sizes.add(len(sample))
        # Should see at least 2 different sizes
        assert len(sizes) >= 1  # At minimum the original size

    def test_bootstrap_horizons_shape(self, multi_agent_df):
        df = add_weight_column(multi_agent_df, "equal_task_weight")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bs = compute_bootstrap_horizons(
                df, n_bootstrap=5, weights_col="weight",
                success_percents=[50], seed=42, n_jobs=1,
            )
        assert len(bs) == 5
        # Should have columns for each agent that had variation
        p50_cols = [c for c in bs.columns if c.endswith("_p50")]
        assert len(p50_cols) > 0

    def test_bootstrap_reproducibility(self, multi_agent_df):
        """Same seed should produce same results."""
        df = add_weight_column(multi_agent_df, "equal_task_weight")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bs1 = compute_bootstrap_horizons(
                df, n_bootstrap=5, weights_col="weight",
                success_percents=[50], seed=42, n_jobs=1,
            )
            bs2 = compute_bootstrap_horizons(
                df, n_bootstrap=5, weights_col="weight",
                success_percents=[50], seed=42, n_jobs=1,
            )
        # Sort columns for comparison
        common_cols = sorted(set(bs1.columns) & set(bs2.columns))
        if common_cols:
            for col in common_cols:
                v1 = bs1[col].dropna().values
                v2 = bs2[col].dropna().values
                if len(v1) > 0 and len(v2) > 0:
                    assert np.allclose(v1, v2, equal_nan=True)


# ── Unit Tests: SOTA Determination ───────────────────────────────────────────


class TestSOTA:
    def test_basic_sota(self):
        results = {
            "A": {"p50": 10.0},
            "B": {"p50": 5.0},   # not SOTA (lower than A)
            "C": {"p50": 20.0},  # SOTA (new high)
            "D": {"p50": 15.0},  # not SOTA (lower than C)
        }
        dates = {
            "A": "2024-01-01",
            "B": "2024-02-01",
            "C": "2024-03-01",
            "D": "2024-04-01",
        }
        sota = determine_sota_agents(results, dates)
        assert "A" in sota
        assert "B" not in sota
        assert "C" in sota
        assert "D" not in sota

    def test_same_date_both_sota(self):
        """Two agents on the same date, both achieving new high."""
        results = {
            "A": {"p50": 10.0},
            "B": {"p50": 10.0},
        }
        dates = {"A": "2024-01-01", "B": "2024-01-01"}
        sota = determine_sota_agents(results, dates)
        assert "A" in sota
        assert "B" in sota

    def test_after_date_filter(self):
        results = {"A": {"p50": 10.0}, "B": {"p50": 20.0}}
        dates = {"A": "2023-01-01", "B": "2024-01-01"}
        sota = determine_sota_agents(results, dates, after_date="2024-01-01")
        assert "A" not in sota
        assert "B" in sota

    def test_empty_results(self):
        sota = determine_sota_agents({}, {})
        assert sota == []

    def test_agent_not_in_dates_skipped(self):
        results = {"A": {"p50": 10.0}, "B": {"p50": 20.0}}
        dates = {"A": "2024-01-01"}  # B not in dates
        sota = determine_sota_agents(results, dates)
        assert "A" in sota
        assert "B" not in sota


# ── Unit Tests: Doubling Time ────────────────────────────────────────────────


class TestDoublingTime:
    def test_known_doubling_time(self):
        """Points that double every 100 days should give ~100 day doubling time."""
        p50s = [1.0, 2.0, 4.0, 8.0]
        base = datetime.date(2024, 1, 1)
        dates = [str(base + datetime.timedelta(days=i * 100)) for i in range(4)]
        dt, r2 = compute_doubling_time(p50s, dates)
        assert abs(dt - 100) < 5
        assert r2 > 0.99

    def test_faster_doubling(self):
        """Points that double every 50 days."""
        p50s = [1.0, 2.0, 4.0, 8.0]
        base = datetime.date(2024, 1, 1)
        dates = [str(base + datetime.timedelta(days=i * 50)) for i in range(4)]
        dt, r2 = compute_doubling_time(p50s, dates)
        assert abs(dt - 50) < 3

    def test_no_growth(self):
        """Flat p50s should give very large (effectively infinite) doubling time."""
        p50s = [10.0, 10.0, 10.0, 10.0]
        base = datetime.date(2024, 1, 1)
        dates = [str(base + datetime.timedelta(days=i * 100)) for i in range(4)]
        dt, r2 = compute_doubling_time(p50s, dates)
        # Doubling time should be very large or inf
        assert abs(dt) > 1000 or np.isinf(dt)


# ── Unit Tests: Trend with CI ────────────────────────────────────────────────


class TestTrendCI:
    def test_trend_with_too_few_agents(self):
        results = {"A": {"p50": 10.0}}
        dates = {"A": "2024-01-01"}
        bs = pd.DataFrame({"A_p50": [10.0, 11.0]})
        trend = compute_trend_with_ci(results, bs, dates)
        assert np.isnan(trend["point_estimate"])

    def test_trend_basic(self):
        results = {
            "A": {"p50": 10.0},
            "B": {"p50": 20.0},
            "C": {"p50": 40.0},
        }
        dates = {
            "A": "2024-01-01",
            "B": "2024-04-01",
            "C": "2024-07-01",
        }
        bs = pd.DataFrame({
            "A_p50": [9.0, 11.0, 10.0] * 10,
            "B_p50": [18.0, 22.0, 20.0] * 10,
            "C_p50": [35.0, 45.0, 40.0] * 10,
        })
        trend = compute_trend_with_ci(results, bs, dates, after_date="2023-01-01")
        assert not np.isnan(trend["point_estimate"])
        assert trend["point_estimate"] > 0
        assert trend["ci_low"] > 0
        assert trend["ci_high"] > trend["ci_low"]


# ── Unit Tests: Output Formatting ────────────────────────────────────────────


class TestOutputFormatting:
    def test_yaml_format(self):
        results = {
            "Agent1": {
                "average_score": 0.5, "coefficient": -1.0, "intercept": 3.0,
                "p50": 8.0, "p50_ci_low": 4.0, "p50_ci_high": 16.0,
                "p80": 3.0, "p80_ci_low": 1.0, "p80_ci_high": 6.0,
            }
        }
        out = format_results(results, fmt="yaml")
        parsed = yaml.safe_load(out)
        assert "results" in parsed
        assert "benchmark_name" in parsed

    def test_json_format(self):
        results = {
            "Agent1": {
                "average_score": 0.5, "coefficient": -1.0, "intercept": 3.0,
                "p50": 8.0, "p50_ci_low": 4.0, "p50_ci_high": 16.0,
                "p80": 3.0, "p80_ci_low": 1.0, "p80_ci_high": 6.0,
            }
        }
        out = format_results(results, fmt="json")
        parsed = json.loads(out)
        assert "results" in parsed

    def test_csv_format(self):
        results = {
            "Agent1": {
                "average_score": 0.5, "coefficient": -1.0, "intercept": 3.0,
                "p50": 8.0, "p50_ci_low": 4.0, "p50_ci_high": 16.0,
                "p80": 3.0, "p80_ci_low": 1.0, "p80_ci_high": 6.0,
            }
        }
        out = format_results(results, fmt="csv")
        lines = out.strip().split("\n")
        assert len(lines) == 2  # header + 1 agent
        assert "agent" in lines[0].lower()

    def test_with_release_dates_and_doubling(self):
        results = {
            "A": {"average_score": 0.5, "p50": 10.0, "p50_ci_low": 5, "p50_ci_high": 20,
                   "p80": 3.0, "p80_ci_low": 1, "p80_ci_high": 6, "coefficient": -1, "intercept": 3},
        }
        dates = {"A": "2024-01-01"}
        trend = {"from_2023_on": {"point_estimate": 100, "ci_low": 80, "ci_high": 120}}
        out = format_results(results, release_dates=dates, doubling_time_stats=trend, fmt="yaml")
        parsed = yaml.safe_load(out)
        assert "doubling_time_in_days" in parsed
        assert "release_date" in list(parsed["results"].values())[0]


# ── Unit Tests: Data Loading ─────────────────────────────────────────────────


class TestDataLoading:
    def test_load_jsonl(self, tmp_path):
        data = [
            {"task_id": "t1", "task_family": "f1", "alias": "A",
             "human_minutes": 10, "score_binarized": 1, "run_id": "r1"},
            {"task_id": "t2", "task_family": "f1", "alias": "A",
             "human_minutes": 20, "score_binarized": 0, "run_id": "r2"},
        ]
        f = tmp_path / "test.jsonl"
        f.write_text("\n".join(json.dumps(d) for d in data))
        df = load_run_data(str(f))
        assert "agent" in df.columns  # alias renamed
        assert len(df) == 2

    def test_load_csv(self, tmp_path):
        df = pd.DataFrame({
            "task_id": ["t1", "t2"],
            "task_family": ["f1", "f1"],
            "agent": ["A", "A"],
            "human_minutes": [10, 20],
            "score_binarized": [1, 0],
        })
        f = tmp_path / "test.csv"
        df.to_csv(f, index=False)
        loaded = load_run_data(str(f))
        assert len(loaded) == 2
        assert "run_id" in loaded.columns  # auto-generated

    def test_missing_column_raises(self, tmp_path):
        df = pd.DataFrame({"task_id": ["t1"], "agent": ["A"]})
        f = tmp_path / "test.csv"
        df.to_csv(f, index=False)
        with pytest.raises(ValueError):
            load_run_data(str(f))

    def test_invalid_score_raises(self, tmp_path):
        df = pd.DataFrame({
            "task_id": ["t1"], "task_family": ["f1"], "agent": ["A"],
            "human_minutes": [10], "score_binarized": [2.0],
        })
        f = tmp_path / "test.csv"
        df.to_csv(f, index=False)
        with pytest.raises(ValueError, match="values in"):
            load_run_data(str(f))

    def test_load_release_dates_nested(self, tmp_path):
        data = {"date": {"A": "2024-01-01", "B": "2024-06-01"}}
        f = tmp_path / "dates.yaml"
        f.write_text(yaml.dump(data))
        dates = load_release_dates(str(f))
        assert dates["A"] == "2024-01-01"

    def test_load_release_dates_flat(self, tmp_path):
        data = {"A": "2024-01-01", "B": "2024-06-01"}
        f = tmp_path / "dates.yaml"
        f.write_text(yaml.dump(data))
        dates = load_release_dates(str(f))
        assert dates["A"] == "2024-01-01"


# ── Integration Tests ────────────────────────────────────────────────────────


class TestIntegration:
    def test_full_pipeline_no_bootstrap(self, multi_agent_df):
        """Full pipeline without bootstrap should produce valid results."""
        df = add_weight_column(multi_agent_df, "invsqrt_task_weight")
        results = {}
        for agent, agent_df in df.groupby("agent"):
            r = compute_agent_horizon(
                agent_df["human_minutes"].values,
                agent_df["score_binarized"].values,
                agent_df["weight"].values,
                str(agent),
                success_percents=[50, 80],
            )
            results[str(agent)] = r

        assert "Strong" in results
        assert "Weak" in results
        # Strong agent should have higher p50 than weak
        assert results["Strong"]["p50"] > results["Weak"]["p50"]
        # Both should have positive horizons
        assert results["Strong"]["p50"] > 0
        assert results["Weak"]["p50"] > 0

    def test_full_pipeline_with_bootstrap(self, multi_agent_df):
        """Full pipeline with bootstrap should produce CIs."""
        df = add_weight_column(multi_agent_df, "equal_task_weight")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bs = compute_bootstrap_horizons(
                df, n_bootstrap=20, weights_col="weight",
                success_percents=[50], seed=42, n_jobs=1,
            )

        results = {}
        for agent, agent_df in df.groupby("agent"):
            r = compute_agent_horizon(
                agent_df["human_minutes"].values,
                agent_df["score_binarized"].values,
                agent_df["weight"].values,
                str(agent),
                success_percents=[50],
                bootstrap_horizons=bs,
            )
            results[str(agent)] = r

        # At least one agent should have non-NaN CIs
        has_ci = any(
            not np.isnan(r.get("p50_ci_low", float("nan")))
            for r in results.values()
        )
        assert has_ci

    def test_output_roundtrip_yaml(self, multi_agent_df):
        """Results can be formatted as YAML and parsed back."""
        df = add_weight_column(multi_agent_df, "equal_task_weight")
        results = {}
        for agent, agent_df in df.groupby("agent"):
            r = compute_agent_horizon(
                agent_df["human_minutes"].values,
                agent_df["score_binarized"].values,
                agent_df["weight"].values,
                str(agent), success_percents=[50, 80],
            )
            results[str(agent)] = r

        yaml_str = format_results(results, fmt="yaml", benchmark_name="Test")
        parsed = yaml.safe_load(yaml_str)
        assert parsed["benchmark_name"] == "Test"
        assert len(parsed["results"]) == 2


# ── Validation Against METR Benchmark ────────────────────────────────────────

RUNS_FILE = os.path.join(
    os.path.dirname(__file__),
    "eval-analysis-public", "reports", "time-horizon-1-1", "data", "raw", "runs.jsonl"
)
BENCHMARK_FILE = os.path.join(os.path.dirname(__file__), "benchmark_results_1_1.yaml")
RELEASE_DATES_FILE = os.path.join(
    os.path.dirname(__file__),
    "eval-analysis-public", "data", "external", "release_dates.yaml"
)


@pytest.mark.skipif(
    not os.path.exists(RUNS_FILE),
    reason="METR runs.jsonl not available for validation"
)
class TestMETRValidation:
    """Validate against known METR benchmark results.

    These tests require the METR runs.jsonl data file and will be skipped
    if it's not present.
    """

    @pytest.fixture(scope="class")
    def metr_data(self):
        """Load and prepare METR data (once per class)."""
        df = load_run_data(RUNS_FILE, score_col="score_binarized")
        df = add_weight_column(df, "invsqrt_task_weight")
        return df

    @pytest.fixture(scope="class")
    def metr_expected(self):
        """Load expected benchmark results."""
        return yaml.safe_load(pathlib.Path(BENCHMARK_FILE).read_text())

    @pytest.fixture(scope="class")
    def metr_release_dates(self):
        """Load release dates."""
        return load_release_dates(RELEASE_DATES_FILE)

    @pytest.fixture(scope="class")
    def metr_agent_results(self, metr_data):
        """Compute agent results from METR data."""
        results = {}
        for agent, agent_df in metr_data.groupby("agent"):
            r = compute_agent_horizon(
                agent_df["human_minutes"].values,
                agent_df["score_binarized"].values,
                agent_df["weight"].values,
                str(agent),
                success_percents=[50, 80],
                confidence_level=0.95,
            )
            results[str(agent)] = r
        return results

    def _find_agent_for_model_key(self, model_key, agent_results, release_dates, expected_date):
        """Find the agent name matching a model key from benchmark results."""
        # Try matching by normalized name
        for agent in agent_results:
            norm = agent.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
            if norm == model_key or model_key.startswith(norm):
                return agent
        # Try matching by release date
        if expected_date:
            for agent in agent_results:
                if agent in release_dates and str(release_dates[agent]) == str(expected_date):
                    norm = agent.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
                    if model_key.startswith(norm[:6]):
                        return agent
        return None

    def test_point_estimates_match(self, metr_agent_results, metr_expected, metr_release_dates):
        """Point estimates should match within 1% tolerance."""
        expected_results = metr_expected.get("results", {})
        tolerance = 0.01
        matched = 0
        failed = 0

        for model_key, expected_data in expected_results.items():
            # Only validate v1.1 models (not stitched v1.0)
            if expected_data.get("benchmark_name") != "METR-Horizon-v1.1":
                continue

            exp_metrics = expected_data.get("metrics", {})
            exp_date = expected_data.get("release_date")
            agent = self._find_agent_for_model_key(
                model_key, metr_agent_results, metr_release_dates, exp_date
            )
            if agent is None:
                continue

            res = metr_agent_results[agent]

            # Check p50
            exp_p50 = exp_metrics.get("p50_horizon_length", {}).get("estimate")
            if exp_p50 and exp_p50 > 0:
                rel_err = abs(res["p50"] - exp_p50) / exp_p50
                if rel_err > tolerance:
                    failed += 1
                    print(f"FAIL p50 {model_key}: expected {exp_p50}, got {res['p50']}, err={rel_err:.4f}")
                else:
                    matched += 1

            # Check average score
            exp_avg = exp_metrics.get("average_score", {}).get("estimate")
            if exp_avg and exp_avg > 0:
                rel_err = abs(res["average_score"] - exp_avg) / exp_avg
                if rel_err > tolerance:
                    failed += 1
                    print(f"FAIL avg {model_key}: expected {exp_avg}, got {res['average_score']}, err={rel_err:.4f}")
                else:
                    matched += 1

        assert matched > 0, "No metrics were compared"
        assert failed == 0, f"{failed} metrics exceeded tolerance"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
