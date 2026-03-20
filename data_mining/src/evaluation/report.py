# src/evaluation/report.py
import pandas as pd

def generate_comparison_report(metrics_dict):
    """
    metrics_dict: {'Model_Name': {'F1': 0.8, 'PR-AUC': 0.85}, ...}
    """
    df_report = pd.DataFrame(metrics_dict).T
    # Xuất ra thư mục outputs
    df_report.to_csv("outputs/tables/model_comparison.csv")
    return df_report