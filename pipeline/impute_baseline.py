"""
Stage 4a — Baseline Imputation
Linear interpolation and forward fill as benchmark methods.
These are the naive approaches KNN should outperform.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def linear_interpolation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct missing values using linear interpolation along the time axis.
    For each tenor, missing values are filled by linear interpolation between
    the nearest preceding and following observed values.

    Args:
        df: DataFrame with simulated missing values.

    Returns:
        DataFrame with missing values filled by linear interpolation.
    """
    logger.info("Applying linear interpolation")
    df_filled = df.copy()

    for col in df.columns:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            df_filled[col] = df[col].interpolate(method="time", limit_direction="both")

    remaining = df_filled.isna().sum().sum()
    logger.info(f"Linear interpolation complete — {remaining} values still missing")
    if remaining > 0:
        logger.warning("Some values could not be interpolated (likely at series boundaries)")
        # Fill any remaining with forward/backward fill
        df_filled = df_filled.ffill().bfill()

    return df_filled


def forward_fill(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct missing values using forward fill (last observation carried forward).
    Simple but commonly used in practice for short gaps.

    Args:
        df: DataFrame with simulated missing values.

    Returns:
        DataFrame with missing values filled by forward fill.
    """
    logger.info("Applying forward fill")
    df_filled = df.ffill().bfill()  # bfill handles any leading NaNs

    remaining = df_filled.isna().sum().sum()
    logger.info(f"Forward fill complete — {remaining} values still missing")
    return df_filled
