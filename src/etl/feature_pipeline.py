from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.config_loader import project_root


ROOT = project_root()
DATA_PATH = ROOT / "data" / "processed" / "ml_ready_dataset.csv"


def load_processed_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.copy()


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["balance_change_percent"] = out["balance_trend"] * 100.0
    out["transaction_decline_percent"] = np.where(
        out["transaction_value_trend"] < 0,
        -out["transaction_value_trend"] * 100.0,
        0.0,
    )
    out["digital_engagement_decline_score"] = np.clip(-out["change_in_digital_activity"], 0.0, 1.5)
    out["service_friction_score"] = (
        out["complaint_occurrence"] * 0.45
        + out["complaint_resolution_time"] / 25.0
        + out["failed_service_transaction_incidents"] * 0.30
        + out["service_channel_wait_time"] / 18.0
        + out["failed_declined_transaction_rate"] * 4.0
    )
    out["financial_stress_index"] = (
        out["loan_repayment_status"] * 2.0
        + out["credit_utilization"]
        + out["recent_overdraft_occurrences"] * 0.5
    )
    out["channel_diversity_score"] = out["number_of_active_banking_channels"] / 6.0
    out["relationship_health_score"] = (
        0.65 * out["income_salary_inflow_regularity"]
        + 0.55 * out["number_of_active_products"] / 8.0
        - 0.75 * out["digital_engagement_decline_score"]
        - 0.60 * out["service_friction_score"]
        - 0.50 * out["financial_stress_index"]
    )
    out["activity_recency_index"] = out["days_since_last_meaningful_interaction"] / 90.0
    out["transaction_consistency_index"] = np.clip(1.0 / (1.0 + out["failed_declined_transaction_rate"] * 10.0), 0.1, 1.0)

    return out


def run_data_quality_checks(df: pd.DataFrame) -> dict:
    checks = {}
    checks["n_rows"] = int(df.shape[0])
    checks["n_columns"] = int(df.shape[1])
    checks["duplicate_customers"] = int(df.duplicated(subset=["customer_id"]).sum())
    checks["missing_values_total"] = int(df.isna().sum().sum())
    checks["missing_by_column"] = df.isna().sum().sort_values(ascending=False).to_dict()
    checks["target_prevalence"] = float(df["relationship_exit"].mean())
    checks["ratio_bounds"] = {
        "inflow_to_outflow_ratio_min": float(df["inflow_to_outflow_ratio"].min()),
        "inflow_to_outflow_ratio_max": float(df["inflow_to_outflow_ratio"].max()),
        "first_contact_resolution_min": float(df["first_contact_resolution"].min()),
        "first_contact_resolution_max": float(df["first_contact_resolution"].max()),
    }
    checks["negative_balance_count"] = int((df["average_account_balance"] < 0).sum())
    return checks


def leakage_guard_checks(df: pd.DataFrame) -> bool:
    forbidden_columns = {
        "future_account_status",
        "future_transaction_count",
        "future_balance",
        "future_exit_signal",
        "future_relationship_exit",
    }
    return not forbidden_columns.intersection(set(df.columns))


def prepare_model_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    engineered = add_engineered_features(df)
    target = engineered["relationship_exit"].copy()

    exclude_columns = {"customer_id", "relationship_exit"}
    feature_columns = [col for col in engineered.columns if col not in exclude_columns]
    feature_frame = engineered[feature_columns].copy()

    categorical_columns = feature_frame.select_dtypes(include=["object", "category"]).columns.tolist()
    if categorical_columns:
        feature_frame = pd.get_dummies(feature_frame, columns=categorical_columns, drop_first=False)
        feature_columns = feature_frame.columns.tolist()

    return feature_frame, target, feature_columns
