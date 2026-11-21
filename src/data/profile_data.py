
from pathlib import Path
import pandas as pd

# Project directories
ROOT = Path(__file__).resolve().parents[2]

EXCEL_FILE = (
    ROOT / "datasets" / "online_retail_II" /
    "raw" / "online_retail_II.xlsx"
)

CSV_FILE = (
    ROOT / "datasets" / "supplier_sales" /
    "raw" / "dataset1.csv"
)

REPORT_DIR = ROOT / "reports" / "profiling"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def profile_dataset(df, dataset_name):

    lines = []

    def add(value=""):
        lines.append(str(value))

    add("=" * 65)
    add(f"DATASET PROFILE: {dataset_name}")
    add("=" * 65)

    # Dataset size
    add(f"\nTotal rows: {df.shape[0]}")
    add(f"Total columns: {df.shape[1]}")

    # Column names
    add("\nCOLUMN NAMES")
    for column in df.columns:
        add(column)

    # Data types
    add("\nDATA TYPES")
    add(df.dtypes.to_string())

    # Missing values
    add("\nMISSING VALUES")
    missing = df.isnull().sum()
    add(missing.to_string())
    add(f"Total missing cells: {df.isnull().sum().sum()}")

    # Duplicate records
    add("\nDUPLICATE RECORDS")
    add(f"Total duplicates: {df.duplicated().sum()}")

    # Numerical information
    add("\nNUMERICAL STATISTICS")
    numerical = df.describe(include="number")
    add(numerical.to_string())

    # Categorical information
    add("\nCATEGORICAL INFORMATION")
    categorical = df.describe(include=["object", "category"])
    add(categorical.to_string())

    # Memory usage
    add("\nMEMORY USAGE")
    add(f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    report = "\n".join(lines)

    print(report)

    # Save report
    safe_name = "".join(
        c if c.isalnum() or c in "_-" else "_"
        for c in dataset_name
    )

    report_path = REPORT_DIR / f"{safe_name}_profile.txt"
    report_path.write_text(report, encoding="utf-8")

    print(f"\nReport saved: {report_path}")

def profile_excel():

    print("\n\nPROFILE 1: ONLINE RETAIL II")

    if not EXCEL_FILE.exists():
        raise FileNotFoundError(EXCEL_FILE)

    excel = pd.ExcelFile(EXCEL_FILE)

    for sheet in excel.sheet_names:

        df = pd.read_excel(
            EXCEL_FILE,
            sheet_name=sheet
        )

        profile_dataset(
            df,
            f"Online_Retail_II_{sheet}"
        )

def profile_csv():

    print("\n\nPROFILE 2: SUPPLIER SALES")

    if not CSV_FILE.exists():
        raise FileNotFoundError(CSV_FILE)

    df = pd.read_csv(CSV_FILE)

    profile_dataset(
        df,
        "Supplier_Sales"
    )

if __name__ == "__main__":

    profile_excel()
    profile_csv()

    print("\n" + "=" * 65)
    print("BOTH DATASETS PROFILED SUCCESSFULLY")
    print("=" * 65)