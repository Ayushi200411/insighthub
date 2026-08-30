import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_science.eda import run_eda, format_eda_report
from data_science.automl import run_automl

def run_data_agent(csv_path, target_col=None):
    report, df = run_eda(csv_path)
    eda_summary = format_eda_report(report)

    if target_col is None:
        return {
            "answer": f"Here's the EDA report:\n\n{eda_summary}\n\n"
                      f"To train a baseline model, tell me which column to predict."
        }

    ml_result = run_automl(csv_path, target_col)
    if "error" in ml_result:
        return {"answer": ml_result["error"]}

    answer = (
        f"EDA Summary:\n{eda_summary}\n\n"
        f"ML Baseline ({ml_result['task_type']}):\n"
        f"  Predicting: {ml_result['target_column']}\n"
        f"  Dropped columns: {ml_result['dropped_columns']}\n"
        f"  Metrics: {ml_result['metrics']}\n"
        f"  Top features: {ml_result['top_features']}"
    )
    return {"answer": answer}