
"""
MARKETMIND AI
STEP 14: ONLINE RETAIL II FEATURE ENGINEERING

Input:
datasets/online_retail_II/raw/online_retail_II.xlsx

Outputs:
1. Transaction features
2. Invoice features
3. Customer RFM features
4. Product features
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ==================================================
# 1. PROJECT PATHS
# ==================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "datasets"
    / "online_retail_II"
    / "raw"
    / "online_retail_II.xlsx"
)

OUTPUT_DIR = (
    ROOT
    / "datasets"
    / "online_retail_II"
    / "processed"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "eda"
    / "online_retail_II"
    / "feature_engineering"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 60)
print("MARKETMIND AI - ONLINE RETAIL FEATURE ENGINEERING")
print("=" * 60)


# ==================================================
# 2. CHECK AND LOAD EXCEL DATASET
# ==================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Excel dataset not found: {INPUT_FILE}"
    )

workbook = pd.ExcelFile(INPUT_FILE)

print("\nAvailable Excel sheets:")
print(workbook.sheet_names)

sheet_name = next(
    (
        sheet for sheet in workbook.sheet_names
        if "2010" in sheet and "2011" in sheet
    ),
    None
)

if sheet_name is None:
    raise ValueError(
        "Could not find the Year 2010-2011 worksheet. "
        f"Available sheets: {workbook.sheet_names}"
    )

df = pd.read_excel(
    INPUT_FILE,
    sheet_name=sheet_name
)

print("\nSelected worksheet:", sheet_name)
print("Original dataset shape:", df.shape)


# ==================================================
# 3. VALIDATE ACTUAL COLUMN NAMES
# ==================================================

required_columns = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country"
]

missing_columns = (
    set(required_columns) - set(df.columns)
)

if missing_columns:
    raise ValueError(
        f"Missing Excel columns: {missing_columns}"
    )

df = df[required_columns].copy()


# ==================================================
# 4. DATA CLEANING
# ==================================================

df["Invoice"] = (
    df["Invoice"].astype("string").str.strip()
)

df["StockCode"] = (
    df["StockCode"].astype("string").str.strip()
)

df["Description"] = (
    df["Description"].fillna("Unknown").astype(str).str.strip()
)

df["Country"] = (
    df["Country"].fillna("Unknown").astype(str).str.strip()
)

df["Quantity"] = pd.to_numeric(
    df["Quantity"], errors="coerce"
)

df["Price"] = pd.to_numeric(
    df["Price"], errors="coerce"
)

df["Customer ID"] = pd.to_numeric(
    df["Customer ID"], errors="coerce"
).astype("Int64")

df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"],
    errors="coerce",
    dayfirst=True
)

df = df.dropna(
    subset=[
        "Invoice",
        "StockCode",
        "Quantity",
        "Price",
        "InvoiceDate"
    ]
).copy()

df = df[df["Invoice"] != ""].copy()

print("\nData cleaning completed.")
print("Cleaned transaction rows:", len(df))


# ==================================================
# 5. TRANSACTION FEATURES
# ==================================================

# Total value of each transaction line

df["LineTotal"] = (
    df["Quantity"] * df["Price"]
)

# Cancellation identification

df["IsCancelledInvoice"] = (
    df["Invoice"].str.upper().str.startswith("C", na=False)
)

# Identify negative quantity return lines
# that are not cancellation invoices

df["IsReturnLine"] = (
    (df["Quantity"] < 0)
    & (~df["IsCancelledInvoice"])
)

# Valid positive sales

df["IsValidSale"] = (
    (~df["IsCancelledInvoice"])
    & (df["Quantity"] > 0)
    & (df["Price"] > 0)
)

# Calendar features

df["Year"] = df["InvoiceDate"].dt.year
df["Month"] = df["InvoiceDate"].dt.month
df["Day"] = df["InvoiceDate"].dt.day

df["Quarter"] = df["InvoiceDate"].dt.quarter

df["Hour"] = df["InvoiceDate"].dt.hour

df["DayOfWeek"] = (
    df["InvoiceDate"].dt.dayofweek
)

df["IsWeekend"] = (
    df["DayOfWeek"] >= 5
)

# Seasonal features

df["MonthSin"] = np.sin(
    2 * np.pi * df["Month"] / 12
)

df["MonthCos"] = np.cos(
    2 * np.pi * df["Month"] / 12
)

print("Transaction features created.")


# ==================================================
# 6. FILTER VALID SALES FOR BUSINESS ANALYSIS
# ==================================================

sales = df[
    df["IsValidSale"]
    & df["Customer ID"].notna()
].copy()

if sales.empty:
    raise ValueError(
        "No valid customer sales found after cleaning."
    )

print("Valid customer sales:", len(sales))


# ==================================================
# 7. INVOICE LEVEL FEATURES
# ==================================================

invoice_features = (
    sales.groupby("Invoice")
    .agg(
        InvoiceTotal=("LineTotal", "sum"),
        InvoiceQuantity=("Quantity", "sum"),
        InvoiceItemCount=("StockCode", "size"),
        InvoiceUniqueProducts=("StockCode", "nunique"),
        InvoiceUniqueCustomers=("Customer ID", "nunique")
    )
    .reset_index()
)

invoice_features.to_csv(
    OUTPUT_DIR / "invoice_features.csv",
    index=False
)

print("Invoice features created.")


# ==================================================
# 8. ADD INVOICE FEATURES TO TRANSACTIONS
# ==================================================

transaction_features = df.merge(
    invoice_features,
    on="Invoice",
    how="left"
)

transaction_features.to_csv(
    OUTPUT_DIR / "online_retail_transactions_features.csv",
    index=False
)

print("Transaction feature dataset saved.")


# ==================================================
# 9. CUSTOMER LEVEL RFM FEATURES
# ==================================================

reference_date = (
    sales["InvoiceDate"].max()
    + pd.Timedelta(days=1)
)

customer_features = (
    sales.groupby("Customer ID")
    .agg(
        LastPurchase=("InvoiceDate", "max"),
        FirstPurchase=("InvoiceDate", "min"),
        Frequency=("Invoice", "nunique"),
        Monetary=("LineTotal", "sum"),
        TotalQuantity=("Quantity", "sum"),
        UniqueProducts=("StockCode", "nunique"),
        UniqueCountries=("Country", "nunique")
    )
    .reset_index()
)

# Recency: days since last purchase

customer_features["Recency"] = (
    reference_date
    - customer_features["LastPurchase"]
).dt.days

# Customer lifetime in days

customer_features["CustomerTenureDays"] = (
    customer_features["LastPurchase"]
    - customer_features["FirstPurchase"]
).dt.days

# Average order value

customer_features["AverageOrderValue"] = (
    customer_features["Monetary"]
    / customer_features["Frequency"]
)

customer_features["AverageQuantityPerOrder"] = (
    customer_features["TotalQuantity"]
    / customer_features["Frequency"]
)

customer_features.to_csv(
    OUTPUT_DIR / "customer_rfm_features.csv",
    index=False
)

print("Customer RFM features created.")


# ==================================================
# 10. PRODUCT LEVEL FEATURES
# ==================================================

product_features = (
    sales.groupby("StockCode")
    .agg(
        ProductDescription=("Description", "first"),
        TotalRevenue=("LineTotal", "sum"),
        TotalUnitsSold=("Quantity", "sum"),
        TotalOrders=("Invoice", "nunique"),
        UniqueCustomers=("Customer ID", "nunique"),
        AverageUnitPrice=("Price", "mean"),
        CountriesSold=("Country", "nunique")
    )
    .reset_index()
)

product_features.to_csv(
    OUTPUT_DIR / "product_features.csv",
    index=False
)

print("Product features created.")


# ==================================================
# 11. GENERATE FEATURE ENGINEERING REPORT
# ==================================================

report = []

report.append("MARKETMIND AI")
report.append("ONLINE RETAIL II FEATURE ENGINEERING REPORT")
report.append("=" * 60)

report.append(f"Excel worksheet: {sheet_name}")
report.append(f"Original rows: {len(df)}")
report.append(f"Valid sales rows: {len(sales)}")

report.append(
    f"Transaction feature columns: {len(transaction_features.columns)}"
)

report.append(
    f"Invoice records: {len(invoice_features)}"
)

report.append(
    f"Customer records: {len(customer_features)}"
)

report.append(
    f"Product records: {len(product_features)}"
)

report.append("\nTRANSACTION FEATURE COLUMNS:")
report.extend(transaction_features.columns.tolist())

report.append("\nCUSTOMER RFM FEATURES:")
report.extend(customer_features.columns.tolist())

report.append("\nPRODUCT FEATURES:")
report.extend(product_features.columns.tolist())

report.append("\nOUTPUT DIRECTORY:")
report.append(str(OUTPUT_DIR))

report.append("\nMISSING VALUES IN TRANSACTION DATA:")
report.append(
    str(transaction_features.isnull().sum())
)

report.append("\nCUSTOMER RFM PREVIEW:")
report.append(str(customer_features.head()))

report.append("\nPRODUCT PREVIEW:")
report.append(str(product_features.head()))

report.append("\nFEATURE ENGINEERING COMPLETED SUCCESSFULLY.")

report_file = (
    REPORT_DIR / "online_retail_feature_summary.txt"
)

report_file.write_text(
    "\n".join(report),
    encoding="utf-8"
)


# ==================================================
# 12. FINAL RESULTS
# ==================================================

print("\n" + "=" * 60)
print("ONLINE RETAIL II FEATURE ENGINEERING COMPLETED")
print("=" * 60)

print("\nTRANSACTION DATA:", transaction_features.shape)
print("INVOICE DATA:", invoice_features.shape)
print("CUSTOMER DATA:", customer_features.shape)
print("PRODUCT DATA:", product_features.shape)

print("\nGenerated files:")

for file in sorted(OUTPUT_DIR.glob("*.csv")):
    print("-", file.name)

print("\nReport:", report_file)

print("\nMARKETMIND AI STEP 14 COMPLETED SUCCESSFULLY")
