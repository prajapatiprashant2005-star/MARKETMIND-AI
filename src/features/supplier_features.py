
"""
MARKETMIND AI
Step 13: Supplier Sales Feature Engineering
"""

from pathlib import Path
import pandas as pd
import numpy as np

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "datasets" / "supplier_sales"
    / "raw" / "dataset1.csv"
)

OUTPUT_DIR = (
    ROOT / "datasets" / "supplier_sales"
    / "processed"
)

REPORT_DIR = (
    ROOT / "reports" / "eda"
    / "supplier_sales"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "supplier_sales_features.csv"
REPORT_FILE = REPORT_DIR / "feature_engineering_summary.txt"


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("\nMARKETMIND AI - SUPPLIER FEATURE ENGINEERING")

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE, low_memory=False)

print("\nOriginal dataset shape:", df.shape)


# --------------------------------------------------
# 3. VALIDATE COLUMNS
# --------------------------------------------------

required_columns = [
    "YEAR",
    "MONTH",
    "SUPPLIER",
    "ITEM CODE",
    "ITEM DESCRIPTION",
    "ITEM TYPE",
    "RETAIL SALES",
    "RETAIL TRANSFERS",
    "WAREHOUSE SALES"
]

missing = set(required_columns) - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df[required_columns].copy()


# --------------------------------------------------
# 4. CLEAN DATA TYPES
# --------------------------------------------------

numeric_columns = [
    "YEAR",
    "MONTH",
    "RETAIL SALES",
    "RETAIL TRANSFERS",
    "WAREHOUSE SALES"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col].astype("string").str.replace(
            ",", "", regex=False
        ).str.replace("$", "", regex=False).str.strip(),
        errors="coerce"
    )

df["SUPPLIER"] = df["SUPPLIER"].fillna("Unknown").astype(str)
df["ITEM CODE"] = df["ITEM CODE"].fillna("Unknown").astype(str)

df["ITEM DESCRIPTION"] = (
    df["ITEM DESCRIPTION"].fillna("Unknown").astype(str)
)

df["ITEM TYPE"] = df["ITEM TYPE"].fillna("Unknown").astype(str)

df = df.dropna(subset=["YEAR", "MONTH"])

df = df[
    df["MONTH"].between(1, 12)
].copy()

df["YEAR"] = df["YEAR"].astype(int)
df["MONTH"] = df["MONTH"].astype(int)

df["DATE"] = pd.to_datetime(
    dict(
        year=df["YEAR"],
        month=df["MONTH"],
        day=1
    ),
    errors="coerce"
)

df = df.dropna(subset=["DATE"])


# --------------------------------------------------
# 5. HANDLE MISSING NUMERIC VALUES
# --------------------------------------------------

sales_columns = [
    "RETAIL SALES",
    "RETAIL TRANSFERS",
    "WAREHOUSE SALES"
]

for col in sales_columns:
    df[col] = df[col].fillna(0)

print("Data cleaning completed.")


# --------------------------------------------------
# 6. MONTHLY SUPPLIER-ITEM AGGREGATION
# --------------------------------------------------

df = (
    df.groupby(
        ["SUPPLIER", "ITEM CODE", "DATE"],
        as_index=False,
        sort=False
    )
    .agg({
        "ITEM DESCRIPTION": "first",
        "ITEM TYPE": "first",
        "RETAIL SALES": "sum",
        "RETAIL TRANSFERS": "sum",
        "WAREHOUSE SALES": "sum"
    })
)

df = df.sort_values(
    ["SUPPLIER", "ITEM CODE", "DATE"]
).reset_index(drop=True)

print("Monthly aggregation completed.")


# --------------------------------------------------
# 7. TIME FEATURES
# --------------------------------------------------

df["YEAR"] = df["DATE"].dt.year
df["MONTH"] = df["DATE"].dt.month
df["QUARTER"] = df["DATE"].dt.quarter

df["MONTH_SIN"] = np.sin(
    2 * np.pi * df["MONTH"] / 12
)

df["MONTH_COS"] = np.cos(
    2 * np.pi * df["MONTH"] / 12
)


# --------------------------------------------------
# 8. SALES RATIO FEATURES
# --------------------------------------------------

df["RETAIL_SALES_RATIO"] = (
    df["RETAIL SALES"] /
    (df["WAREHOUSE SALES"] + 1)
)

df["TRANSFER_TO_RETAIL_RATIO"] = (
    df["RETAIL TRANSFERS"] /
    (df["RETAIL SALES"] + 1)
)


# --------------------------------------------------
# 9. HISTORICAL LAG FEATURES
# --------------------------------------------------

group_columns = ["SUPPLIER", "ITEM CODE"]

for col in sales_columns:

    df[f"{col}_LAG_1"] = (
        df.groupby(group_columns)[col].shift(1)
    )

    df[f"{col}_LAG_2"] = (
        df.groupby(group_columns)[col].shift(2)
    )


# --------------------------------------------------
# 10. ROLLING AVERAGE FEATURES
# --------------------------------------------------

for col in ["RETAIL SALES", "WAREHOUSE SALES"]:

    df[f"{col}_ROLLING_3"] = (
        df.groupby(group_columns)[col]
        .transform(
            lambda s: s.shift(1).rolling(
                window=3,
                min_periods=1
            ).mean()
        )
    )


# --------------------------------------------------
# 11. SAVE FEATURE DATASET
# --------------------------------------------------

df.to_csv(OUTPUT_FILE, index=False)


# --------------------------------------------------
# 12. GENERATE REPORT
# --------------------------------------------------

report = []

report.append("MARKETMIND AI")
report.append("SUPPLIER SALES FEATURE ENGINEERING REPORT")
report.append("=" * 55)

report.append(f"Final rows: {len(df)}")
report.append(f"Final columns: {len(df.columns)}")

report.append("\nFEATURE COLUMNS:")

for col in df.columns:
    report.append(col)

report.append("\nMISSING VALUES:")
report.append(str(df.isnull().sum()))

report.append("\nDATASET PREVIEW:")
report.append(str(df.head()))

report.append("\nOUTPUT FILE:")
report.append(str(OUTPUT_FILE))

REPORT_FILE.write_text(
    "\n".join(report),
    encoding="utf-8"
)


# --------------------------------------------------
# 13. FINAL STATUS
# --------------------------------------------------

print("\n" + "=" * 55)
print("SUPPLIER FEATURE ENGINEERING COMPLETED")
print("=" * 55)

print("Final dataset shape:", df.shape)
print("Features created:", len(df.columns))

print("\nSaved dataset:")
print(OUTPUT_FILE)

print("\nSaved report:")
print(REPORT_FILE)

print("\nFirst five records:")
print(df.head())

print("\nMARKETMIND AI STEP 13 COMPLETED SUCCESSFULLY")
