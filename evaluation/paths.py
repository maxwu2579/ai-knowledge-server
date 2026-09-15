"""Shared repository paths for offline evaluation scripts."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = PROJECT_ROOT / "evaluation"
EVALUATION_DATA_DIR = EVALUATION_ROOT / "data"
EVALUATION_REPORTS_DIR = EVALUATION_ROOT / "reports"
CHROMA_DATA_ROOT = PROJECT_ROOT / "data"
