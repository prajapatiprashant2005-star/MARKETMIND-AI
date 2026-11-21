
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

EXCEL_FILE = (
    ROOT / "datasets" / "online_retail_II" /
    "processed" / "online_retail_II_cleaned.xlsx"
)

CSV_FILE = (
    ROOT / "datasets" / "supplier_sales" /
    "processed" / "supplier_sales_cleaned.csv"
)

RETAIL_REPORT = ROOT / "reports" / "eda" / "online_retail_II"
SUPPLIER_REPORT = ROOT / "reports" / "eda" / "supplier_sales"

for folder in [RETAIL_REPORT, SUPPLIER_REPORT]:
    (folder / "charts").mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


# ==========================================
# 2. SAFE FILE NAMES
# ==========================================

def safe_name(value):
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value))[:70]


# ==========================================
# 3. GENERAL EDA FUNCTION
# ==========================================

def analyze_dataset(df, dataset_name, output_folder):

    print("\n" + "=" * 65)
    print("EDA:", dataset_name)
    print("=" * 65)

    # Remove completely empty rows for analysis only
    df = df.dropna(how="all")

    numeric = df.select_dtypes(include="number")
    categorical = df.select_dtypes(
        include=["object", "category", "bool"]
    )

    # Basic information
    print("\nDATASET INFORMATION")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Memory MB:", round(
        df.memory_usage(deep=True).sum() / 1024**2, 2
    ))

    print("\nCOLUMN NAMES")
    print(df.columns.tolist())

    # Missing values
    missing = pd.DataFrame({
        "Column": df.columns,
        "Missing_Count": df.isna().sum().values,
        "Missing_Percentage": (
            df.isna().mean().values * 100
        ).round(2)
    })

    missing.to_csv(
        output_folder / "missing_values.csv",
        index=False
    )

    # Numerical analysis
    if not numeric.empty:

        numerical_stats = numeric.describe().T

        numerical_stats.to_csv(
            output_folder / "numerical_statistics.csv"
        )

        print("\nNUMERICAL STATISTICS")
        print(numerical_stats)

        # Correlation
        correlation = numeric.corr()

        correlation.to_csv(
            output_folder / "correlation_matrix.csv"
        )

        # Correlation heatmap
        if len(numeric.columns) >= 2:

            corr = numeric.corr()

            plt.figure(figsize=(12, 8))

            sns.heatmap(
                corr,
                cmap="coolwarm",
                center=0,
                annot=False
            )

            plt.title(dataset_name + " - Correlation Heatmap")
            plt.tight_layout()

            plt.savefig(
                output_folder / "charts" / "correlation_heatmap.png",
                dpi=150
            )

            plt.close()

        # Histograms and boxplots
        for column in numeric.columns[:10]:

            clean_values = numeric[column].dropna()

            if clean_values.empty:
                continue

            plt.figure(figsize=(9, 5))

            plt.hist(
                clean_values.sample(
                    min(len(clean_values), 50000),
                    random_state=42
                ),
                bins=40,
                edgecolor="black"
            )

            plt.title(f"Distribution of {column}")
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.tight_layout()

            plt.savefig(
                output_folder / "charts" /
                f"{safe_name(column)}_histogram.png",
                dpi=150
            )

            plt.close()

            plt.figure(figsize=(9, 4))

            plt.boxplot(clean_values.sample(
                min(len(clean_values), 50000),
                random_state=42
            ))

            plt.title(f"Outlier Inspection: {column}")
            plt.ylabel(column)
            plt.tight_layout()

            plt.savefig(
                output_folder / "charts" /
                f"{safe_name(column)}_boxplot.png",
                dpi=150
            )

            plt.close()

    # Categorical analysis
    if not categorical.empty:

        categorical_summary = []

        for column in categorical.columns:

            counts = df[column].value_counts(dropna=False)

            categorical_summary.append({
                "Column": column,
                "Unique_Values": df[column].nunique(),
                "Most_Common_Value": str(counts.index[0])
                if len(counts) else "",
                "Most_Common_Count": int(counts.iloc[0])
                if len(counts) else 0
            })

            # Top 10 values chart
            top_values = df[column].value_counts().head(10)

            if not top_values.empty:

                plt.figure(figsize=(10, 5))

                top_values.sort_values().plot(kind="barh")

                plt.title(f"Top Values: {column}")
                plt.xlabel("Count")
                plt.tight_layout()

                plt.savefig(
                    output_folder / "charts" /
                    f"{safe_name(column)}_top_values.png",
                    dpi=150
                )

                plt.close()

        pd.DataFrame(categorical_summary).to_csv(
            output_folder / "categorical_summary.csv",
            index=False
        )

    # Final summary
    summary = [
        f"DATASET: {dataset_name}",
        "=" * 60,
        f"Total rows: {len(df)}",
        f"Total columns: {len(df.columns)}",
        f"Numerical columns: {len(numeric.columns)}",
        f"Categorical columns: {len(categorical.columns)}",
        f"Missing cells: {df.isna().sum().sum()}",
        f"Duplicate rows: {df.duplicated().sum()}",
        "",
        "COLUMN TYPES:",
        df.dtypes.to_string(),
        "",
        "MISSING VALUES:",
        df.isna().sum().to_string()
    ]

    (output_folder / "eda_summary.txt").write_text(
        "\n".join(summary),
        encoding="utf-8"
    )

    print("\nEDA completed for:", dataset_name)
    print("Reports saved in:", output_folder)


# ==========================================
# 4. ONLINE RETAIL II EXCEL ANALYSIS
# ==========================================

def analyze_excel():

    print("\nONLINE RETAIL II ANALYSIS")

    if not EXCEL_FILE.exists():
        raise FileNotFoundError(EXCEL_FILE)

    excel = pd.ExcelFile(EXCEL_FILE)

    for sheet in excel.sheet_names:

        print("\nProcessing sheet:", sheet)

        df = pd.read_excel(
            EXCEL_FILE,
            sheet_name=sheet
        )

        folder = RETAIL_REPORT / safe_name(sheet)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "charts").mkdir(exist_ok=True)

        analyze_dataset(
            df,
            f"Online Retail II - {sheet}",
            folder
        )


# ==========================================
# 5. SUPPLIER SALES CSV ANALYSIS
# ==========================================

def analyze_csv():

    print("\nSUPPLIER SALES ANALYSIS")

    if not CSV_FILE.exists():
        raise FileNotFoundError(CSV_FILE)

    df = pd.read_csv(CSV_FILE)

    analyze_dataset(
        df,
        "Supplier Sales",
        SUPPLIER_REPORT
    )


# ==========================================
# 6. MAIN PROGRAM
# ==========================================

if __name__ == "__main__":

    analyze_excel()

    analyze_csv()

    print("\n" + "=" * 65)
    print("MARKETMIND AI EDA COMPLETED SUCCESSFULLY")
    print("=" * 65)
