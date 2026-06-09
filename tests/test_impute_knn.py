"""
Tests for pipeline/impute_knn.py
"""

import numpy as np
import pandas as pd
import pytest

from pipeline.impute_knn import impute, evaluate_k_values


def make_clean_df(n_dates: int = 100, seed: int = 42) -> pd.DataFrame:
    """
    Create a synthetic Treasury-like DataFrame for testing.
    Smooth, correlated rates across tenors — realistic curve shape.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2008-01-01", periods=n_dates)

    # Base rate — random walk
    base = np.cumsum(rng.normal(0, 0.02, n_dates)) + 3.0

    # Tenors add a spread over the base — realistic curve shape
    spreads = {"DGS1MO": -0.5, "DGS3MO": -0.4, "DGS6MO": -0.3, "DGS1": -0.2,
               "DGS2": 0.0, "DGS3": 0.1, "DGS5": 0.3, "DGS7": 0.5,
               "DGS10": 0.7, "DGS20": 1.0, "DGS30": 1.1}

    data = {}
    for tenor, spread in spreads.items():
        noise = rng.normal(0, 0.03, n_dates)
        data[tenor] = np.clip(base + spread + noise, 0.01, 10.0)

    return pd.DataFrame(data, index=dates)


def make_gaps(df: pd.DataFrame, pct: float = 0.10, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Introduce random missing values into a clean DataFrame."""
    rng = np.random.default_rng(seed)
    df_gaps = df.copy()
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)

    for col in df.columns:
        n = int(len(df) * pct)
        idx = rng.choice(len(df), size=n, replace=False)
        mask.iloc[idx, mask.columns.get_loc(col)] = True
        df_gaps.iloc[idx, df_gaps.columns.get_loc(col)] = np.nan

    return df_gaps, mask


class TestImpute:
    def test_no_missing_values_after_imputation(self):
        df_clean = make_clean_df()
        df_gaps, _ = make_gaps(df_clean)
        df_filled = impute(df_gaps, k=5)
        assert df_filled.isna().sum().sum() == 0

    def test_output_shape_unchanged(self):
        df_clean = make_clean_df()
        df_gaps, _ = make_gaps(df_clean)
        df_filled = impute(df_gaps, k=5)
        assert df_filled.shape == df_gaps.shape

    def test_output_index_unchanged(self):
        df_clean = make_clean_df()
        df_gaps, _ = make_gaps(df_clean)
        df_filled = impute(df_gaps, k=5)
        assert list(df_filled.index) == list(df_gaps.index)

    def test_observed_values_preserved(self):
        """Values that were not masked should be unchanged after imputation."""
        df_clean = make_clean_df()
        df_gaps, mask = make_gaps(df_clean, pct=0.10)
        df_filled = impute(df_gaps, k=5)

        # Where mask is False (not removed), values should be very close to original
        for col in df_clean.columns:
            not_masked = ~mask[col]
            original = df_clean.loc[not_masked, col].values
            filled = df_filled.loc[not_masked, col].values
            np.testing.assert_allclose(original, filled, atol=0.01)

    def test_reconstructed_values_in_plausible_range(self):
        """Reconstructed values should stay within the observed range of the series."""
        df_clean = make_clean_df()
        df_gaps, mask = make_gaps(df_clean)
        df_filled = impute(df_gaps, k=5)

        for col in df_clean.columns:
            col_min = df_clean[col].min() - 0.5  # small buffer
            col_max = df_clean[col].max() + 0.5
            reconstructed = df_filled.loc[mask[col], col]
            assert (reconstructed >= col_min).all(), f"{col}: reconstructed values below minimum"
            assert (reconstructed <= col_max).all(), f"{col}: reconstructed values above maximum"

    def test_different_k_values_produce_results(self):
        """Pipeline should run cleanly for all supported K values."""
        df_clean = make_clean_df()
        df_gaps, _ = make_gaps(df_clean)
        for k in [3, 5, 7, 10]:
            df_filled = impute(df_gaps, k=k)
            assert df_filled.isna().sum().sum() == 0, f"Missing values remain for k={k}"

    def test_reconstruction_accuracy_within_threshold(self):
        """On clean synthetic data, MAE should be within a reasonable threshold."""
        df_clean = make_clean_df(n_dates=200)
        df_gaps, mask = make_gaps(df_clean, pct=0.10)
        df_filled = impute(df_gaps, k=5)

        actual = df_clean.values[mask.values]
        predicted = df_filled.values[mask.values]
        mae_bp = np.mean(np.abs(actual - predicted)) * 100

        # On smooth synthetic data, MAE should be well under 20bp
        assert mae_bp < 20.0, f"MAE too high: {mae_bp:.2f}bp"


class TestEvaluateKValues:
    def test_returns_dataframe_with_correct_columns(self):
        df_clean = make_clean_df()
        df_gaps, mask = make_gaps(df_clean)
        results = evaluate_k_values(df_gaps, df_clean, mask, k_values=[3, 5])
        assert "k" in results.columns
        assert "mae_bp" in results.columns
        assert "rmse_bp" in results.columns

    def test_returns_one_row_per_k_value(self):
        df_clean = make_clean_df()
        df_gaps, mask = make_gaps(df_clean)
        k_values = [3, 5, 7]
        results = evaluate_k_values(df_gaps, df_clean, mask, k_values=k_values)
        assert len(results) == len(k_values)

    def test_mae_is_positive(self):
        df_clean = make_clean_df()
        df_gaps, mask = make_gaps(df_clean)
        results = evaluate_k_values(df_gaps, df_clean, mask, k_values=[5])
        assert (results["mae_bp"] >= 0).all()

    def test_rmse_greater_than_or_equal_to_mae(self):
        """RMSE should always be >= MAE (penalises large errors more)."""
        df_clean = make_clean_df()
        df_gaps, mask = make_gaps(df_clean)
        results = evaluate_k_values(df_gaps, df_clean, mask, k_values=[5])
        assert (results["rmse_bp"] >= results["mae_bp"]).all()
