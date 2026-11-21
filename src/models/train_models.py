
"""
MARKETMIND AI
STEP 15: MACHINE LEARNING MODEL TRAINING

Dataset 1: Supplier Sales CSV
Model: Random Forest Regressor

Dataset 2: Online Retail II Excel
Model: K-Means Customer Segmentation

The datasets are processed separately.
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

SUPPLIER_FILE = (
    ROOT / "datasets" / "supplier_sales" / "raw" / "dataset1.csv"
)

RETAIL_FILE = (
    ROOT / "datasets" / "online_retail_II" / "raw"
    / "online_retail_II.xlsx"
)

MODEL_DIR = ROOT / "models" / "trained"

SUPPLIER_REPORT = (
    ROOT / "reports" / "models" / "supplier_sales"
)

RETAIL_REPORT = (
    ROOT / "reports" / "models" / "online_retail"
)

for folder in [MODEL_DIR, SUPPLIER_REPORT, RETAIL_REPORT]:
    folder.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. SUPPLIER SALES PREDICTION
# --------------------------------------------------

def train_supplier_model():

    print("\n" + "=" * 60)
    print("MODEL 1: SUPPLIER SALES PREDICTION")
    print("=" * 60)

    if not SUPPLIER_FILE.exists():
        raise FileNotFoundError(
            f"Supplier CSV not found: {SUPPLIER_FILE}"
        )

    df = pd.read_csv(SUPPLIER_FILE, low_memory=False)

    # Standardize column names
    df.columns = df.columns.str.strip().str.upper()

    required = [
        "YEAR",
        "MONTH",
        "SUPPLIER",
        "RETAIL SALES",
        "RETAIL TRANSFERS",
        "WAREHOUSE SALES"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing supplier columns: {missing}"
        )

    # Convert numeric columns
    numeric_columns = [
        "YEAR",
        "MONTH",
        "RETAIL SALES",
        "RETAIL TRANSFERS",
        "WAREHOUSE SALES"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["SUPPLIER"] = (
        df["SUPPLIER"].fillna("UNKNOWN").astype(str)
    )

    df = df.dropna(subset=numeric_columns)

    # Aggregate supplier sales by month
    data = df.groupby(
        ["YEAR", "MONTH", "SUPPLIER"],
        as_index=False
    ).agg({
        "RETAIL SALES": "sum",
        "RETAIL TRANSFERS": "sum",
        "WAREHOUSE SALES": "sum"
    })

    data = data[
        (data["MONTH"] >= 1) &
        (data["MONTH"] <= 12)
    ].copy()

    if len(data) < 20:
        raise ValueError(
            "Not enough valid supplier records for training."
        )

    data["PERIOD"] = pd.to_datetime(
        data["YEAR"].astype(int).astype(str)
        + "-"
        + data["MONTH"].astype(int).astype(str)
        + "-01"
    )

    # Create machine learning features
    X = data[
        [
            "YEAR",
            "MONTH",
            "SUPPLIER",
            "RETAIL TRANSFERS",
            "WAREHOUSE SALES"
        ]
    ].copy()

    X["SUPPLIER"] = X["SUPPLIER"].astype(str)

    X = pd.get_dummies(
        X,
        columns=["SUPPLIER"],
        dtype=int
    )

    y = data["RETAIL SALES"]

    # Time-based train/test split
    unique_periods = sorted(data["PERIOD"].unique())

    if len(unique_periods) < 2:
        raise ValueError(
            "Supplier data needs at least two different months."
        )

    split_index = max(
        1,
        min(
            len(unique_periods) - 1,
            int(len(unique_periods) * 0.8)
        )
    )

    split_date = unique_periods[split_index]

    train_mask = data["PERIOD"] < split_date
    test_mask = data["PERIOD"] >= split_date

    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]

    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(
            "Invalid time-based train/test split."
        )

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    # Train model
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=15,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    # Save trained model and its feature names
    joblib.dump(
        {
            "model": model,
            "feature_columns": list(X.columns)
        },
        MODEL_DIR / "supplier_sales_model.pkl"
    )

    results = pd.DataFrame({
        "Actual_Sales": y_test.values,
        "Predicted_Sales": predictions
    })

    results.to_csv(
        SUPPLIER_REPORT / "supplier_predictions.csv",
        index=False
    )

    metrics = {
        "model": "Random Forest Regressor",
        "target": "RETAIL SALES",
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }

    with open(
        SUPPLIER_REPORT / "model_metrics.json",
        "w"
    ) as file:
        json.dump(metrics, file, indent=4)

    print("\nSUPPLIER MODEL RESULTS")
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 4))

    print("Supplier model saved successfully.")


# --------------------------------------------------
# 3. ONLINE RETAIL CUSTOMER SEGMENTATION
# --------------------------------------------------

def train_retail_model():

    print("\n" + "=" * 60)
    print("MODEL 2: ONLINE RETAIL CUSTOMER SEGMENTATION")
    print("=" * 60)

    if not RETAIL_FILE.exists():
        raise FileNotFoundError(
            f"Online Retail Excel file not found: {RETAIL_FILE}"
        )

    # Select the required worksheet
    excel = pd.ExcelFile(RETAIL_FILE)

    sheet_name = (
        "Year 2010-2011"
        if "Year 2010-2011" in excel.sheet_names
        else excel.sheet_names[0]
    )

    print("Selected worksheet:", sheet_name)

    df = pd.read_excel(
        RETAIL_FILE,
        sheet_name=sheet_name
    )

    df.columns = df.columns.str.strip()

    required = [
        "Invoice",
        "Quantity",
        "InvoiceDate",
        "Price",
        "Customer ID"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing Online Retail columns: {missing}"
        )

    # Clean data
    df["Quantity"] = pd.to_numeric(
        df["Quantity"], errors="coerce"
    )

    df["Price"] = pd.to_numeric(
        df["Price"], errors="coerce"
    )

    df["Customer ID"] = pd.to_numeric(
        df["Customer ID"], errors="coerce"
    )

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], errors="coerce"
    )

    df = df.dropna(
        subset=[
            "Customer ID",
            "Invoice",
            "InvoiceDate",
            "Quantity",
            "Price"
        ]
    )

    # Remove returns and invalid transactions
    df = df[
        (df["Quantity"] > 0) &
        (df["Price"] > 0)
    ].copy()

    if df.empty:
        raise ValueError(
            "No valid retail transactions found."
        )

    df["Customer ID"] = (
        df["Customer ID"].astype("int64").astype(str)
    )

    df["Invoice"] = df["Invoice"].astype(str)

    df["TotalAmount"] = df["Quantity"] * df["Price"]

    reference_date = (
        df["InvoiceDate"].max() + pd.Timedelta(days=1)
    )

    # Calculate RFM features
    rfm = df.groupby("Customer ID").agg(
        Recency=(
            "InvoiceDate",
            lambda x: (reference_date - x.max()).days
        ),
        Frequency=("Invoice", "nunique"),
        Monetary=("TotalAmount", "sum")
    ).reset_index()

    rfm = rfm.replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    if len(rfm) < 4:
        raise ValueError(
            "At least four customers are required for segmentation."
        )

    # Log transformation reduces extreme value impact
    features = np.log1p(
        rfm[["Recency", "Frequency", "Monetary"]]
    )

    # Scale and cluster
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        (
            "kmeans",
            KMeans(
                n_clusters=4,
                random_state=42,
                n_init=10
            )
        )
    ])

    rfm["Cluster"] = pipeline.fit_predict(features)

    # Rank clusters by average monetary value
    cluster_values = (
        rfm.groupby("Cluster")["Monetary"]
        .mean()
        .sort_values(ascending=False)
    )

    names = [
        "Premium Customers",
        "Loyal Customers",
        "Regular Customers",
        "Low Value Customers"
    ]

    segment_map = {
        cluster: names[index]
        for index, cluster in enumerate(cluster_values.index)
    }

    rfm["Customer_Segment"] = (
        rfm["Cluster"].map(segment_map)
    )

    # Save customer segments
    rfm.to_csv(
        RETAIL_REPORT / "customer_segments.csv",
        index=False
    )

    # Save model and metadata
    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_names": [
                "Recency",
                "Frequency",
                "Monetary"
            ],
            "segment_names": segment_map
        },
        MODEL_DIR / "online_retail_segmentation_model.pkl"
    )

    # Save segment summary
    summary = (
        rfm.groupby("Customer_Segment")
        .agg(
            Customers=("Customer ID", "count"),
            Average_Recency=("Recency", "mean"),
            Average_Frequency=("Frequency", "mean"),
            Average_Monetary=("Monetary", "mean")
        )
        .reset_index()
    )

    summary.to_csv(
        RETAIL_REPORT / "segment_summary.csv",
        index=False
    )

    print("\nCUSTOMER SEGMENTATION RESULTS")
    print("Total customers:", len(rfm))
    print("Total segments:", rfm["Cluster"].nunique())
    print(summary.to_string(index=False))

    print("\nOnline Retail model saved successfully.")


# --------------------------------------------------
# 4. MAIN EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("          MARKETMIND AI")
    print("       MACHINE LEARNING SYSTEM")
    print("=" * 60)

    try:

        train_supplier_model()

        train_retail_model()

        print("\n" + "=" * 60)
        print("BOTH MACHINE LEARNING MODELS TRAINED SUCCESSFULLY")
        print("=" * 60)

    except Exception as error:

        print("\nMODEL TRAINING FAILED")
        print("Error:", error)
        raise
