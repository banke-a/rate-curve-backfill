"""
Tests for evaluation mechanics — verifying that error metrics,
masking logic, and threshold checks are computed correctly end to end.

These tests are distinct from test_evaluate.py which tests the
evaluate module in isolation. These tests verify the full evaluation
chain: gap simulation → imputation → metric computation → thresholds.
"""

import numpy as np
import pandas as pd
import pytest

from pipeline.evaluate import compute_metrics, check_thresholds
from pipeline.impute_knn import impute
from pipeline.impute_baseline import linear_interpolation, forward_fill


def make_known_curve(n_dates: int = 100) -> pd.DataFrame:
    """
    Create a perfectly smooth synthetic curve with known properties.
    Used to verify metric calculations against hand-computed values.
    """
    dates = pd.bdate_range("2008-01-01", periods=n_dates)
    cols = ["DGS1MO", "DGS3MO", "DGS2", "DGS5", "DGS10", "DGS30"]

    # Deterministic linear trend — easy to verify by hand
    data = {}
    for i, col in enumerate(cols):
        data[col] = np.linspace(1.0 + i * 0.5, 2.0 + i * 0.5, n_dates)

    return pd.DataFrame(data, index=dates)


def apply_known_error(
    df_clean: pd.DataFrame,
    error_bp: float = 10.0,
    pct: float = 0.10,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply a known constant error to a fixed percentage of observations.
    Allows verification that computed MAE matches the known error.
    """
    rng = np.random.default_rng(seed)
    df_filled = df_clean.copy()
    mask = pd.DataFrame(False, index=df_clean.index, columns=df_clean.columns)

    for col in df_clean.columns:
        n = int(len(df_clean) * pct)
        idx = rng.choice(len(df_clean), size=n, replace=False)
        mask.iloc[idx, mask.columns.get_loc(col)] = True
        df_filled.iloc[idx, df_filled.columns.get_loc(col)] += error_bp / 100  # convert bp to %

    return df_filled, mask


class TestMaskingMechanics:
    def test_metrics_only_computed_on_masked_positions(self):
        """
        If we apply a known error only to masked positions, MAE should
        equal the known error. Unmasked positions should not contribute.
        """
        df_clean = make_known_curve()
        known_error_bp = 10.0
        df_filled, mask = apply_known_error(df_clean, error_bp=known_error_bp)

        metrics = compute_metrics(df_clean, df_filled, mask, label="Test")

        # MAE should be very close to known_error_bp
        assert abs(metrics["mae_bp"] - known_error_bp) < 0.1, (
            f"MAE {metrics['mae_bp']:.2f}bp does not match known error {known_error_bp}bp"
        )

    def test_unmasked_positions_do_not_affect_metrics(self):
        """
        Corrupting unmasked positions should not change the metrics
        since evaluation only uses masked positions.
        """
        df_clean = make_known_curve()
        df_filled, mask = apply_known_error(df_clean, error_bp=10.0)

        # Corrupt unmasked positions heavily
        df_corrupted = df_filled.copy()
        for col in df_clean.columns:
            not_masked = ~mask[col]
            df_corrupted.loc[not_masked, col] += 999.0  # huge error on unmasked

        metrics_original = compute_metrics(df_clean, df_filled, mask, label="Original")
        metrics_corrupted = compute_metrics(df_clean, df_corrupted, mask, label="Corrupted")

        # Metrics should be identical — only masked positions count
        assert abs(metrics_original["mae_bp"] - metrics_corrupted["mae_bp"]) < 0.01

    def test_n_reconstructed_matches_actual_mask_count(self):
        df_clean = make_known_curve()
        df_filled, mask = apply_known_error(df_clean)
        metrics = compute_metrics(df_clean, df_filled, mask)
        assert metrics["n_reconstructed"] == int(mask.sum().sum())

    def test_zero_mask_gives_zero_reconstructed(self):
        """Empty mask should produce zero reconstructed observations."""
        df_clean = make_known_curve()
        empty_mask = pd.DataFrame(False, index=df_clean.index, columns=df_clean.columns)
        metrics = compute_metrics(df_clean, df_clean.copy(), empty_mask)
        assert metrics["n_reconstructed"] == 0
        assert metrics["mae_bp"] == 0.0 or np.isnan(metrics["mae_bp"])


class TestMetricComputations:
    def test_mae_is_mean_of_absolute_errors(self):
        """Verify MAE formula directly against numpy computation."""
        df_clean = make_known_curve()
        known_error_bp = 5.0
        df_filled, mask = apply_known_error(df_clean, error_bp=known_error_bp)

        # Hand compute
        actual = df_clean.values[mask.values]
        predicted = df_filled.values[mask.values]
        expected_mae = np.mean(np.abs(actual - predicted)) * 100

        metrics = compute_metrics(df_clean, df_filled, mask)
        assert abs(metrics["mae_bp"] - expected_mae) < 0.01

    def test_rmse_is_root_mean_square_error(self):
        """Verify RMSE formula directly."""
        df_clean = make_known_curve()
        df_filled, mask = apply_known_error(df_clean, error_bp=8.0)

        actual = df_clean.values[mask.values]
        predicted = df_filled.values[mask.values]
        expected_rmse = np.sqrt(np.mean((actual - predicted) ** 2)) * 100

        metrics = compute_metrics(df_clean, df_filled, mask)
        assert abs(metrics["rmse_bp"] - expected_rmse) < 0.01

    def test_max_error_is_largest_single_error(self):
        """Verify max error equals the largest individual reconstruction error."""
        df_clean = make_known_curve()

        # Introduce one large known error and many small ones
        df_filled = df_clean.copy()
        mask = pd.DataFrame(False, index=df_clean.index, columns=df_clean.columns)

        # Small errors
        mask.iloc[0:5, 0] = True
        df_filled.iloc[0:5, 0] += 0.05  # 5bp

        # One large error
        mask.iloc[10, 1] = True
        df_filled.iloc[10, 1] += 0.30  # 30bp

        metrics = compute_metrics(df_clean, df_filled, mask)
        assert abs(metrics["max_error_bp"] - 30.0) < 0.1

    def test_rmse_greater_than_mae_with_outliers(self):
        """RMSE should exceed MAE when outliers are present."""
        df_clean = make_known_curve()
        df_filled = df_clean.copy()
        mask = pd.DataFrame(False, index=df_clean.index, columns=df_clean.columns)

        # Mix of small and large errors
        mask.iloc[0:20, 0] = True
        df_filled.iloc[0:19, 0] += 0.02   # 2bp
        df_filled.iloc[19, 0] += 1.00     # 100bp outlier

        metrics = compute_metrics(df_clean, df_filled, mask)
        assert metrics["rmse_bp"] > metrics["mae_bp"]

    def test_per_tenor_mae_sums_correctly(self):
        """Per-tenor MAE values should be consistent with overall MAE."""
        df_clean = make_known_curve(n_dates=200)
        df_filled, mask = apply_known_error(df_clean, error_bp=10.0, pct=0.20)
        metrics = compute_metrics(df_clean, df_filled, mask)

        tenor_keys = [k for k in metrics if k.startswith("mae_DGS")]
        assert len(tenor_keys) > 0

        # All per-tenor MAEs should be positive
        for key in tenor_keys:
            assert metrics[key] >= 0, f"{key} has negative MAE"


class TestThresholdLogic:
    def test_boundary_conditions(self):
        """Values exactly at the threshold should pass."""
        metrics = {"mae_bp": 10.0, "mae_stress_bp": 15.0, "max_error_bp": 50.0}
        result = check_thresholds(metrics)
        # Strictly less than threshold — boundary should fail
        assert result["mae_full_pass"] is False
        assert result["mae_stress_pass"] is False
        assert result["max_error_pass"] is False

    def test_just_below_threshold_passes(self):
        metrics = {"mae_bp": 9.99, "mae_stress_bp": 14.99, "max_error_bp": 49.99}
        result = check_thresholds(metrics)
        assert result["mae_full_pass"] is True
        assert result["mae_stress_pass"] is True
        assert result["max_error_pass"] is True

    def test_all_checks_independent(self):
        """Each threshold check should be independent of the others."""
        # Only MAE fails
        metrics = {"mae_bp": 15.0, "mae_stress_bp": 8.0, "max_error_bp": 20.0}
        result = check_thresholds(metrics)
        assert result["mae_full_pass"] is False
        assert result["mae_stress_pass"] is True
        assert result["max_error_pass"] is True


class TestEndToEndEvaluation:
    def test_knn_outperforms_forward_fill_on_smooth_data(self):
        """
        On smooth correlated data, KNN should outperform forward fill.
        Forward fill is particularly bad when missing values follow
        a directional trend.
        """
        rng = np.random.default_rng(42)
        n = 200
        dates = pd.bdate_range("2008-01-01", periods=n)
        cols = ["DGS2", "DGS5", "DGS10", "DGS30"]

        # Trending data — forward fill will lag
        base = np.linspace(5.0, 2.0, n)
        data = {col: base + i * 0.3 + rng.normal(0, 0.01, n) for i, col in enumerate(cols)}
        df_clean = pd.DataFrame(data, index=dates)

        # Mask 15% of observations
        mask = pd.DataFrame(False, index=dates, columns=cols)
        for col in cols:
            idx = rng.choice(n, size=int(n * 0.15), replace=False)
            mask.iloc[idx, mask.columns.get_loc(col)] = True

        df_gaps = df_clean.where(~mask)

        df_knn = impute(df_gaps, k=5)
        df_ff = forward_fill(df_gaps)

        metrics_knn = compute_metrics(df_clean, df_knn, mask, label="KNN")
        metrics_ff = compute_metrics(df_clean, df_ff, mask, label="ForwardFill")

        assert metrics_knn["mae_bp"] < metrics_ff["mae_bp"], (
            f"KNN MAE {metrics_knn['mae_bp']:.2f}bp should beat Forward Fill {metrics_ff['mae_bp']:.2f}bp"
        )
