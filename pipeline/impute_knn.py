"""
Stage 4b — KNN Imputation
Reconstruct missing rate observations using K-Nearest Neighbours.

Design:
- Feature space: all available tenor observations on a given date
- Distance metric: Euclidean on z-score normalised rates
- Weighting: inverse distance (closer neighbours contribute more)
- K: evaluated across 3, 5, 7, 10 — default 5

Why KNN for rate curves:
Linear interpolation assumes stable curve shapes. During stress periods
this breaks down. KNN asks: on days when the observable tenors looked
similar to today, what did the missing tenor do? It reconstructs from
the actual comovement structure that existed during the crisis itself.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

DEFAULT_K = 5
K_VALUES_TO_EVALUATE = [3, 5, 7, 10]


def impute(df: pd.DataFrame, k: int = DEFAULT_K) -> pd.DataFrame:
    """
    Reconstruct missing values using KNN imputation.

    The full date x tenor matrix is passed to KNNImputer. For each missing
    value, the K most similar rows (dates) are found using Euclidean distance
    on the available features (tenors). Missing values are filled using the
    inverse-distance-weighted mean of the K neighbours' values for that tenor.

    Args:
        df: DataFrame with simulated missing values.
        k: Number of nearest neighbours (default 5).

    Returns:
        DataFrame with missing values reconstructed by KNN.
    """
    logger.info(f"Applying KNN imputation (k={k})")
    missing_count = df.isna().sum().sum()
    logger.info(f"Missing values to reconstruct: {missing_count:,}")

    # Normalise — z-score per tenor so distance is shape-sensitive not level-sensitive
    scaler = StandardScaler()
    df_values = df.values.copy()

    # Fit scaler on non-missing values only
    col_means = np.nanmean(df_values, axis=0)
    col_stds = np.nanstd(df_values, axis=0)
    col_stds[col_stds == 0] = 1  # Prevent division by zero

    df_normalised = (df_values - col_means) / col_stds

    # KNN imputation on normalised data
    imputer = KNNImputer(n_neighbors=k, weights="distance")
    df_imputed_normalised = imputer.fit_transform(df_normalised)

    # Reverse normalisation
    df_imputed_values = (df_imputed_normalised * col_stds) + col_means

    df_filled = pd.DataFrame(
        df_imputed_values,
        index=df.index,
        columns=df.columns
    )

    remaining = df_filled.isna().sum().sum()
    logger.info(f"KNN imputation complete (k={k}) — {remaining} values still missing")
    return df_filled


def evaluate_k_values(
    df_gaps: pd.DataFrame,
    df_clean: pd.DataFrame,
    mask: pd.DataFrame,
    k_values: list[int] = K_VALUES_TO_EVALUATE,
) -> pd.DataFrame:
    """
    Evaluate KNN reconstruction across multiple K values.
    Returns a summary DataFrame of MAE and RMSE per K value.

    Args:
        df_gaps: DataFrame with simulated missing values.
        df_clean: Original clean DataFrame (ground truth).
        mask: Boolean mask — True where values were removed.
        k_values: List of K values to evaluate.

    Returns:
        DataFrame with columns [k, mae, rmse] — one row per K value.
    """
    logger.info(f"Evaluating K values: {k_values}")
    results = []

    for k in k_values:
        df_filled = impute(df_gaps, k=k)

        # Compare only on masked positions
        actual = df_clean.values[mask.values]
        predicted = df_filled.values[mask.values]

        mae = np.mean(np.abs(actual - predicted)) * 100  # in basis points
        rmse = np.sqrt(np.mean((actual - predicted) ** 2)) * 100  # in basis points

        results.append({"k": k, "mae_bp": round(mae, 2), "rmse_bp": round(rmse, 2)})
        logger.info(f"  K={k}: MAE={mae:.2f}bp, RMSE={rmse:.2f}bp")

    return pd.DataFrame(results)
