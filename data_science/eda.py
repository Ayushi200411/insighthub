import pandas as pd

def run_eda(csv_path):
    df = pd.read_csv(csv_path)

    report = {}
    report["shape"] = {"rows": df.shape[0], "columns": df.shape[1]}
    report["column_types"] = df.dtypes.astype(str).to_dict()
    report["missing_values"] = df.isnull().sum().to_dict()
    report["missing_values"] = {k: v for k, v in report["missing_values"].items() if v > 0}

    numeric_df = df.select_dtypes(include="number")
    if not numeric_df.empty:
        report["summary_stats"] = numeric_df.describe().to_dict()
        if numeric_df.shape[1] > 1:
            report["correlations"] = numeric_df.corr().round(2).to_dict()
    else:
        report["summary_stats"] = {}
        report["correlations"] = {}

    categorical_df = df.select_dtypes(include="object")
    report["categorical_columns"] = list(categorical_df.columns)

    return report, df

def format_eda_report(report):
    lines = []
    lines.append(f"Dataset shape: {report['shape']['rows']} rows, {report['shape']['columns']} columns")
    lines.append(f"\nColumn types:")
    for col, dtype in report["column_types"].items():
        lines.append(f"  {col}: {dtype}")

    if report["missing_values"]:
        lines.append(f"\nMissing values:")
        for col, count in report["missing_values"].items():
            lines.append(f"  {col}: {count} missing")
    else:
        lines.append(f"\nNo missing values found.")

    if report["categorical_columns"]:
        lines.append(f"\nCategorical columns: {', '.join(report['categorical_columns'])}")

    return "\n".join(lines)

if __name__ == "__main__":
    report, df = run_eda("data/titanic.csv")
    print(format_eda_report(report))