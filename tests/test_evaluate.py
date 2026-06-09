"""
Tests for pipeline/evaluate.py
"""

import numpy as np
import pandas as pd
import pytest

from pipeline.evaluate import compute_metrics, check_thresholds


def make_test_data(n_dates: int = 100, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create clean DataFrame, a filled DataFrame with controlled errors,
    and a mask for testing.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2008-01-01", periods=n_dates)
    cols = ["DGS1MO", "DGS3MO", "DGS2", "DGS5", "DGS10", "DGS30"]

    # Clean data
    data = {}
    base = np.cumsum(rng.normal(0, 0.02, n_dates)) + 3.0
    for i, col in enumerate(cols):
        data[col] = np.clip(base + i * 0.2 + rng.normal(0, 0.02, n_dates), 0.01, 10.0)
    df_clean = pd.DataFrame(data, index=dates)

    # Mask — 10% random
    mask = pd.DataFrame(False, index=dates, columns=cols)
    for col in cols:
        idx = rng.choice(n_dates, size=int(n_dates * 0.10), replace=False)
        mask.iloc[idx, mask.columns.get_loc(col)] = True

    # Filled with small controlled errors (3bp)
    df_filled = df_clean.copy()
    for col in cols:
        masked_idx = mask[col]
        df_filled.loc[masked_idx, col] += rng.normal(0, 0.03, masked_idx.sum())

    return df_clean, df_filled, mask


class TestComputeMetrics:
    def test_returns_required_keys(self):
        df_clean, df_filled, mask = make_test_data()
        metrics = compute_metrics(df_clean, df_filled, mask, label="Test")
        required = ["method", "mae_bp", "rmse_bp", "max_error_bp",
                    "mae_stress_bp", "n_reconstructed"]
        for key in required:
            assert key in metrics, f"Missing key: {key}"

    def test_mae_is_positive(self):
        df_clean, df_filled, mask = make_test_data()
        metrics = compute_metrics(df_clean, df_filled, mask)
        assert metrics["mae_bp"] >= 0

    def test_rmse_gte_mae(self):
        df_clean, df_filled, mask = make_test_data()
        metrics = compute_metrics(df_clean, df_filled, mask)
        assert metrics["rmse_bp"] >= metrics["mae_bp"]

    def test_perfect_reconstruction_gives_zero_error(self):
        """If filled equals clean, all errors should be zero."""
        df_clean, _, mask = make_test_data()
        metrics = compute_metrics(df_clean, df_clean.copy(), mask, label="Perfect")
        assert metrics["mae_bp"] == 0.0
        assert metrics["rmse_bp"] == 0.0
        assert metrics["max_error_bp"] == 0.0

    def test_method_label_stored(self):
        df_clean, df_filled, mask = make_test_data()
        metrics = compute_metrics(df_clean, df_filled, mask, label="KNN")
        assert metrics["method"] == "KNN"

    def test_n_reconstructed_matches_mask(self):
        df_clean, df_filled, mask = make_test_data()
        metrics = compute_metrics(df_clean, df_filled, mask)
        assert metrics["n_reconstructed"] == int(mask.sum().sum())

    def test_per_tenor_mae_present(self):
        df_clean, df_filled, mask = make_test_data()
        metrics = compute_metrics(df_clean, df_filled, mask)
        # At least some per-tenor MAE keys should be present
        tenor_keys = [k for k in metrics if k.startswith("mae_DGS")]
        assert len(tenor_keys) > 0

    def test_small_errors_give_low_mae(self):
        """Controlled 3bp errors should produce MAE well under 10bp."""
        df_clean, df_filled, mask = make_test_data()
        metrics = compute_metrics(df_clean, df_filled, mask)
        assert metrics["mae_bp"] < 10.0, f"MAE unexpectedly high: {metrics['mae_bp']}"


class TestCheckThresholds:
    def test_all_pass_when_below_thresholds(self):
        metrics = {
            "mae_bp": 4.0,
            "mae_stress_bp": 8.0,
            "max_error_bp": 20.0,
        }
        result = check_thresholds(metrics)
        assert result["mae_full_pass"] is True
        assert result["mae_stress_pass"] is True
        assert result["max_error_pass"] is True

    def test_mae_fail_when_above_threshold(self):
        metrics = {
            "mae_bp": 15.0,
            "mae_stress_bp": 8.0,
            "max_error_bp": 20.0,
        }
        result = check_thresholds(metrics)
        assert result["mae_full_pass"] is False

    def test_stress_mae_fail_when_above_threshold(self):
        metrics = {
            "mae_bp": 4.0,
            "mae_stress_bp": 20.0,
            "max_error_bp": 20.0,
        }
        result = check_thresholds(metrics)
        assert result["mae_stress_pass"] is False

    def test_max_error_fail_when_above_threshold(self):
        metrics = {
            "mae_bp": 4.0,
            "mae_stress_bp": 8.0,
            "max_error_bp": 60.0,
        }
        result = check_thresholds(metrics)
        assert result["max_error_pass"] is False

    def test_none_stress_mae_handled(self):
        """If stress MAE is None, threshold check should return None not crash."""
        metrics = {
            "mae_bp": 4.0,
            "mae_stress_bp": None,
            "max_error_bp": 20.0,
        }
        result = check_thresholds(metrics)
        assert result["mae_stress_pass"] is None
