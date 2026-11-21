
from pathlib import Path
import pandas as pd

# ==========================================
# 1. PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

EXCEL_FILE = (
    ROOT / "datasets" / "online_retail_II" /
    "raw" / "online_retail_II.xlsx"
)

CSV_FILE = (
    ROOT / "datasets" / "supplier_sales" /
    "raw" / "dataset1.csv"
)

EXCEL_OUTPUT = (
    ROOT / "datasets" / "online_retail_II" /
    "processed" / "online_retail_II_cleaned.xlsx"
)

CSV_OUTPUT = (
    ROOT / "datasets" / "supplier_sales" /
    "processed" / "supplier_sales_cleaned.csv"
)

REPORT_DIR = ROOT / "reports" / "cleaning"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. GENERAL CLEANING FUNCTION
# ==========================================

def clean_dataframe(df, dataset_name):

    original_rows = len(df)
    original_columns = len(df.columns)

    # Remove completely empty rows and columns
    df = df.dropna(how="all")
    df = df.dropna(axis=1, how="all")

    # Clean column names
    df.columns = [
        str(col).strip() for col in df.columns
    ]

    # Remove leading/trailing spaces from text
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].map(
            lambda value: value.strip()
            if isinstance(value, str)
            else value
        )

    # Remove duplicate rows
    duplicates_removed = int(df.duplicated().sum())
    df = df.drop_duplicates()

    # Generate report
    report = [
        f"DATA CLEANING REPORT: {dataset_name}",
        "=" * 55,
        f"Original rows: {original_rows}",
        f"Original columns: {original_columns}",
        f"Final rows: {len(df)}",
        f"Final columns: {len(df.columns)}",
        f"Duplicate rows removed: {duplicates_removed}",
        f"Missing cells remaining: {df.isna().sum().sum()}",
        "",
        "MISSING VALUES BY COLUMN:",
        df.isna().sum().to_string(),
        "",
        "FINAL COLUMN NAMES:",
        str(df.columns.tolist())
    ]

    report_text = "\n".join(report)

    safe_name = "".join(
        c if c.isalnum() or c in "_-" else "_"
        for c in dataset_name
    )

    report_path = REPORT_DIR / f"{safe_name}_cleaning.txt"
    report_path.write_text(report_text, encoding="utf-8")

    print(report_text)
    print(f"\nReport saved: {report_path}")

    return df


# ==========================================
# 3. CLEAN EXCEL DATASET
# ==========================================

def clean_excel():

    print("\n" + "=" * 60)
    print("CLEANING ONLINE RETAIL II EXCEL DATASET")
    print("=" * 60)

    if not EXCEL_FILE.exists():
        raise FileNotFoundError(EXCEL_FILE)

    excel = pd.ExcelFile(EXCEL_FILE)

    cleaned_sheets = {}

    for sheet in excel.sheet_names:

        print(f"\nProcessing Excel sheet: {sheet}")

        df = pd.read_excel(
            EXCEL_FILE,
            sheet_name=sheet
        )

        cleaned_df = clean_dataframe(
            df,
            f"Online_Retail_II_{sheet}"
        )

        cleaned_sheets[sheet] = cleaned_df

    with pd.ExcelWriter(EXCEL_OUTPUT, engine="openpyxl") as writer:

        for sheet, df in cleaned_sheets.items():

            safe_sheet = str(sheet)[:31]

            df.to_excel(
                writer,
                sheet_name=safe_sheet,
                index=False
            )

    print("\nCleaned Excel saved:", EXCEL_OUTPUT)


# ==========================================
# 4. CLEAN CSV DATASET
# ==========================================

def clean_csv():

    print("\n" + "=" * 60)
    print("CLEANING SUPPLIER SALES CSV DATASET")
    print("=" * 60)

    if not CSV_FILE.exists():
        raise FileNotFoundError(CSV_FILE)

    df = pd.read_csv(CSV_FILE)

    cleaned_df = clean_dataframe(
        df,
        "Supplier_Sales"
    )

    cleaned_df.to_csv(
        CSV_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nCleaned CSV saved:", CSV_OUTPUT)


# ==========================================
# 5. MAIN PROGRAM
# ==========================================

if __name__ == "__main__":

    clean_excel()

    clean_csv()

    print("\n" + "=" * 60)
    print("BOTH DATASETS CLEANED SUCCESSFULLY")
    print("=" * 60)
