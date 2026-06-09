"""
run.py — Pipeline entry point
Runs the full rate curve backfilling pipeline.

Usage:
    Full pipeline:
        python run.py

    Ingest only:
        python run.py --ingest-only

    Through gap analysis:
        python run.py --analyse-only

    Use cached data (skip FRED API call):
        python run.py --use-cache
"""

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from pipeline import ingest, validate, analyse_gaps

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
PLOTS_DIR = f"{OUTPUT_DIR}/plots"


def main():
    parser = argparse.ArgumentParser(description="Rate Curve Backfilling Pipeline")
    parser.add_argument("--ingest-only", action="store_true", help="Run ingestion stage only")
    parser.add_argument("--analyse-only", action="store_true", help="Run through gap analysis stage")
    parser.add_argument("--use-cache", action="store_true", help="Use cached FRED data if available")
    args = parser.parse_args()

    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)

    # Stage 1 — Ingest
    logger.info("Stage 1 — Ingest")
    cache = CACHE_PATH if args.use_cache or Path(CACHE_PATH).exists() else None
    df_raw = ingest.load(cache_path=cache)

    if not Path(CACHE_PATH).exists():
        df_raw.to_csv(CACHE_PATH)
        logger.info(f"Data cached to {CACHE_PATH}")

    if args.ingest_only:
        logger.info(f"Data shape: {df_raw.shape}")
        logger.info(f"Date range: {df_raw.index.min().date()} to {df_raw.index.max().date()}")
        logger.info(f"Missing values per tenor:\n{df_raw.isna().sum().to_string()}")
        return

    # Stage 2 — Validate
    logger.info("Stage 2 — Validate")
    result = validate.validate(df_raw)
    df_clean = result["df"]
    logger.info(f"Clean business days: {len(df_clean)}")

    # Stage 3 — Analyse gaps and simulate missing data
    logger.info("Stage 3 — Gap simulation")
    df_gaps, mask = analyse_gaps.simulate_gaps(df_clean)

    logger.info("Generating visualisations")
    analyse_gaps.plot_missing_heatmap(df_gaps, PLOTS_DIR)
    analyse_gaps.plot_curve_shapes(df_clean, PLOTS_DIR)

    logger.info(f"\nGap simulation summary:")
    logger.info(f"  Total observations: {df_clean.size:,}")
    logger.info(f"  Masked observations: {mask.sum().sum():,} ({mask.sum().sum()/df_clean.size:.1%})")
    logger.info(f"  Missing per tenor:\n{df_gaps.isna().sum().to_string()}")

    if args.analyse_only:
        logger.info("Analyse-only mode — stopping here")
        logger.info(f"Plots saved to {PLOTS_DIR}")
        return

    logger.info("Further pipeline stages coming soon")


if __name__ == "__main__":
    main()
