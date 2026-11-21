
import joblib
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "trained" / "supplier_sales_model.pkl"

saved = joblib.load(MODEL_PATH)
model = saved["model"]
features = saved["feature_columns"]

print("Model:", type(model).__name__)
print("Total features:", len(features))
print("First 20 features:", features[:20])

print("\nChecking model input requirements...")

print("Expected input count:",
      getattr(model, "n_features_in_", "Unknown"))

print("Model inspection successful.")
