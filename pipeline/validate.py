"""
Stage 2 — Validate
Schema and quality checks on the raw Treasury DataFrame.
Identifies structural gaps (holidays), outliers, and stale prices.
"""

import logging

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check

logger = logging.getLogger(__name__)

TENOR_COLUMNS = [
    "DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2",
    "DGS3", "DGS5", "DGS7", "DGS10", "DGS20", "DGS30"
]

# Rates must be between 0 and 20% — anything outside is a data error
schema = DataFrameSchema(
    columns={
        col: Column(
            float,
            checks=[
                Check.greater_than_or_equal_to(0),
                Check.less_than_or_equal_to(20),
            ],
            nullable=True,
        )
        for col in TENOR_COLUMNS
    },
    coerce=True,
    strict=False,
)

# Maximum plausible single-day move per tenor (basis points)
MAX_DAILY_MOVE_BP = 100


def validate(df: pd.DataFrame) -> dict:
    """
    Run validation checks on the raw Treasury DataFrame.

    Args:
        df: Raw DataFrame from ingest stage.

    Returns:
        Dictionary with validated DataFrame and quality report.
    """
    logger.info("Running schema validation")
    report = {}

    # Schema check
    try:
        schema.validate(df, lazy=True)
        logger.info("Schema validation passed")
        report["schema_valid"] = True
    except pa.errors.SchemaErrors as e:
        logger.warning(f"Schema violations found:\n{e.failure_cases}")
        report["schema_valid"] = False
        report["schema_errors"] = e.failure_cases.to_dict()

    # Structural gaps — holidays (all tenors missing on same date)
    all_missing = df[df.isna().all(axis=1)]
    report["holiday_dates"] = len(all_missing)
    logger.info(f"Structural gaps (holidays): {len(all_missing)} dates")

    # Business day DataFrame — drop holidays
    df_business = df.dropna(how="all").copy()
    report["business_days"] = len(df_business)

    # Partial missing — some but not all tenors missing on a date
    partial_missing = df_business[df_business.isna().any(axis=1)]
    report["partial_missing_dates"] = len(partial_missing)
    if len(partial_missing) > 0:
        logger.warning(f"Partial missing dates (some tenors missing): {len(partial_missing)}")
        logger.warning(f"  Dates: {partial_missing.index.date.tolist()}")

    # Outlier detection — daily moves exceeding threshold
    daily_moves = df_business.diff().abs() * 100  # convert to basis points
    outliers = (daily_moves > MAX_DAILY_MOVE_BP)
    outlier_count = outliers.sum().sum()
    report["outlier_moves"] = int(outlier_count)
    if outlier_count > 0:
        logger.warning(f"Large daily moves (>{MAX_DAILY_MOVE_BP}bp): {outlier_count} observations")
        for col in TENOR_COLUMNS:
            if col in outliers.columns and outliers[col].any():
                dates = df_business.index[outliers[col]].date.tolist()
                logger.warning(f"  {col}: {dates}")

    # Stale price detection — same value for 5+ consecutive days
    stale_count = 0
    for col in TENOR_COLUMNS:
        if col in df_business.columns:
            rolling_same = (df_business[col] == df_business[col].shift(1))
            consecutive = rolling_same.groupby(
                (rolling_same != rolling_same.shift()).cumsum()
            ).cumsum()
            stale = (consecutive >= 4).sum()
            if stale > 0:
                stale_count += stale
                logger.warning(f"Stale prices in {col}: {stale} observations")

    report["stale_prices"] = stale_count

    # Summary
    logger.info(
        f"Validation complete — {report['business_days']} business days, "
        f"{report.get('outlier_moves', 0)} large moves, "
        f"{report.get('stale_prices', 0)} stale prices"
    )

    return {"df": df_business, "report": report}
