
from pathlib import Path
import pandas as pd

# Project root directory
ROOT = Path(__file__).resolve().parents[2]

# Dataset locations
EXCEL_FILE = (
    ROOT / "datasets" / "online_retail_II" /
    "raw" / "online_retail_II.xlsx"
)

CSV_FILE = (
    ROOT / "datasets" / "supplier_sales" /
    "raw" / "dataset1.csv"
)


def load_online_retail():

    print("\n" + "=" * 60)
    print("ONLINE RETAIL II - EXCEL DATASET")
    print("=" * 60)

    if not EXCEL_FILE.exists():
        raise FileNotFoundError(f"Excel file missing: {EXCEL_FILE}")

    workbook = pd.ExcelFile(EXCEL_FILE)

    print("Available Excel sheets:", workbook.sheet_names)

    for sheet in workbook.sheet_names:

        df = pd.read_excel(EXCEL_FILE, sheet_name=sheet)

        print(f"\nSheet Name: {sheet}")
        print("Dataset Shape:", df.shape)

        print("\nColumn Names:")
        print(df.columns.tolist())

        print("\nFirst 5 Records:")
        print(df.head())

    return workbook.sheet_names


def load_supplier_sales():

    print("\n" + "=" * 60)
    print("SUPPLIER SALES - CSV DATASET")
    print("=" * 60)

    if not CSV_FILE.exists():
        raise FileNotFoundError(f"CSV file missing: {CSV_FILE}")

    df = pd.read_csv(CSV_FILE)

    print("Dataset Shape:", df.shape)

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nFirst 5 Records:")
    print(df.head())

    return df


if __name__ == "__main__":

    load_online_retail()

    supplier_data = load_supplier_sales()

    print("\n" + "=" * 60)
    print("BOTH DATASETS LOADED SUCCESSFULLY")
    print("=" * 60)
