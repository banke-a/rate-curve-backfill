"""
Stage 1 — Ingest
Pull US Treasury yield curve rate series from FRED for the GFC stress window.
"""

import logging
import os
import ssl
import certifi
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

load_dotenv()

logger = logging.getLogger(__name__)

# FRED series IDs for US Treasury yield curve
TREASURY_SERIES = {
    "DGS1MO": "1 Month",
    "DGS3MO": "3 Month",
    "DGS6MO": "6 Month",
    "DGS1":   "1 Year",
    "DGS2":   "2 Year",
    "DGS3":   "3 Year",
    "DGS5":   "5 Year",
    "DGS7":   "7 Year",
    "DGS10":  "10 Year",
    "DGS20":  "20 Year",
    "DGS30":  "30 Year",
}

DEFAULT_START = os.getenv("START_DATE", "2007-01-01")
DEFAULT_END = os.getenv("END_DATE", "2009-12-31")


def load(
    start_date: str = DEFAULT_START,
    end_date: str = DEFAULT_END,
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Pull all Treasury tenor series from FRED and return as a single DataFrame.

    Args:
        start_date: Start of the GFC window (default 2007-01-01)
        end_date: End of the GFC window (default 2009-12-31)
        cache_path: Optional path to cache raw data as CSV (avoids repeated API calls)

    Returns:
        DataFrame with date index and one column per tenor series ID.
    """
    # Return cached data if available
    if cache_path and Path(cache_path).exists():
        logger.info(f"Loading cached data from {cache_path}")
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} tenors from cache")
        return df

    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise ValueError(
            "FRED_API_KEY not found. Set it in your .env file. "
            "Get a free key at https://fred.stlouisfed.org"
        )

    fred = Fred(api_key=api_key)
    series = {}

    logger.info(f"Pulling {len(TREASURY_SERIES)} Treasury series from FRED ({start_date} to {end_date})")

    for series_id, label in TREASURY_SERIES.items():
        try:
            data = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
            series[series_id] = data
            logger.info(f"  {series_id} ({label}): {len(data):,} observations, {data.isna().sum()} missing")
        except Exception as e:
            logger.warning(f"  {series_id} ({label}): failed — {e}")

    df = pd.DataFrame(series)
    df.index.name = "date"
    df = df.sort_index()

    logger.info(f"Ingestion complete — {len(df):,} dates, {len(df.columns)} tenors")
    logger.info(f"Total missing values: {df.isna().sum().sum():,} ({df.isna().mean().mean():.1%} of all observations)")

    # Cache to disk if path provided
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path)
        logger.info(f"Cached raw data to {cache_path}")

    return df
