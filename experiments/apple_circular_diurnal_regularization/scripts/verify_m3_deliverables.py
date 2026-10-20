"""Verify Milestone 3 benchmark deliverables."""
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
csv_path = PROJECT_DIR / "results" / "downstream_classification_metrics.csv"
parquet_path = PROJECT_DIR / "results" / "downstream_test_predictions.parquet"

assert csv_path.exists(), f"Missing {csv_path}"
assert parquet_path.exists(), f"Missing {parquet_path}"

df_metrics = pd.read_csv(csv_path)
print("Metrics CSV rows:", len(df_metrics))
print(df_metrics[["representation", "category", "name", "occupancy_tier", "test_auroc", "delta_auroc", "ci_lower_95", "ci_upper_95", "p_value"]])

df_pred = pd.read_parquet(parquet_path)
print("\nPredictions Parquet shape:", df_pred.shape)
print("Predictions columns:", df_pred.columns.tolist())
assert len(df_pred) == 1671, f"Expected 1671 test predictions, got {len(df_pred)}"
assert not df_pred.isna().any().any(), "Found NaNs in predictions parquet!"

print("\nVerification checks passed successfully!")
