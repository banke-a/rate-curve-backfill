"""
run.py — Pipeline entry point
Runs the full rate curve backfilling pipeline.

Usage:
    Full pipeline:
        python run.py

    Ingest only (useful for exploring data):
        python run.py --ingest-only

    Use cached data (skip FRED API call):
        python run.py --use-cache
"""

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from pipeline import ingest

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run")

DATA_DIR = os.getenv("DATA_DIR", "data/raw")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
CACHE_PATH = f"{DATA_DIR}/treasury_gfc_raw.csv"


def main():
    parser = argparse.ArgumentParser(description="Rate Curve Backfilling Pipeline")
    parser.add_argument("--ingest-only", action="store_true", help="Run ingestion stage only")
    parser.add_argument("--use-cache", action="store_true", help="Use cached FRED data if available")
    args = parser.parse_args()

    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    logger.info("Stage 1 — Ingest")
    cache = CACHE_PATH if args.use_cache else None
    df = ingest.load(cache_path=cache)

    logger.info(f"\nData shape: {df.shape}")
    logger.info(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
    logger.info(f"\nMissing values per tenor:\n{df.isna().sum().to_string()}")

    if args.ingest_only:
        logger.info("Ingest-only mode — stopping here")
        # Cache for subsequent runs
        df.to_csv(CACHE_PATH)
        logger.info(f"Data cached to {CACHE_PATH}")
        return

    logger.info("Further pipeline stages coming soon")


if __name__ == "__main__":
    main()
