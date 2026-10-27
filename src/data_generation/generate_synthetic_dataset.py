from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "config.yaml"
RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "results"

FEATURE_COLUMNS: List[str] = [
    "age_band",
    "customer_tenure",
    "customer_segment",
    "number_of_active_products",
    "primary_account_type",
    "relationship_status",
    "average_account_balance",
    "balance_trend",
    "income_salary_inflow_regularity",
    "loan_repayment_status",
    "credit_utilization",
    "recent_overdraft_occurrences",
    "transaction_count",
    "average_transaction_value",
    "transaction_value_trend",
    "inflow_to_outflow_ratio",
    "recurring_payment_activity",
    "failed_declined_transaction_rate",
    "days_since_last_meaningful_interaction",
    "digital_banking_activity_level",
    "change_in_digital_activity",
    "number_of_active_banking_channels",
    "branch_service_interaction_frequency",
    "unusual_reduction_in_account_usage",
    "complaint_occurrence",
    "complaint_resolution_time",
    "first_contact_resolution",
    "service_interaction_frequency",
    "failed_service_transaction_incidents",
    "service_channel_wait_time",
]


def load_config() -> Dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def ensure_directories() -> None:
    for path in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def generate_customer_profiles(n_customers: int, rng: np.random.Generator) -> pd.DataFrame:
    age_band = rng.choice(
        ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
        size=n_customers,
        p=[0.08, 0.27, 0.29, 0.2, 0.11, 0.05],
    )
    customer_tenure = np.clip(rng.gamma(shape=2.2, scale=4.5, size=n_customers), 0.5, 30.0)
    customer_segment = rng.choice(
        ["Mass", "Mass Affluent", "Premium", "SME", "Wealth"],
        size=n_customers,
        p=[0.43, 0.26, 0.12, 0.15, 0.04],
    )
    primary_account_type = rng.choice(
        ["Current", "Savings", "Business", "Private"],
        size=n_customers,
        p=[0.52, 0.28, 0.12, 0.08],
    )
    relationship_status = rng.choice(["Primary", "Secondary"], size=n_customers, p=[0.82, 0.18])
    number_of_active_products = rng.poisson(lam=2.8, size=n_customers) + 1
    number_of_active_products = np.clip(number_of_active_products, 1, 8)

    profile_df = pd.DataFrame(
        {
            "customer_id": np.arange(1, n_customers + 1),
            "age_band": age_band,
            "customer_tenure": customer_tenure.round(2),
            "customer_segment": customer_segment,
            "number_of_active_products": number_of_active_products,
            "primary_account_type": primary_account_type,
            "relationship_status": relationship_status,
        }
    )
    return profile_df


def generate_monthly_history(profile_df: pd.DataFrame, months: int, rng: np.random.Generator) -> pd.DataFrame:
    rows: List[Dict] = []

    segment_multiplier = {
        "Mass": 0.92,
        "Mass Affluent": 1.08,
        "Premium": 1.25,
        "SME": 1.05,
        "Wealth": 1.42,
    }
    age_multiplier = {
        "18-25": 0.82,
        "26-35": 0.96,
        "36-45": 1.08,
        "46-55": 1.12,
        "56-65": 1.0,
        "65+": 0.9,
    }
    account_multiplier = {
        "Current": 1.12,
        "Savings": 0.88,
        "Business": 1.34,
        "Private": 1.56,
    }

    for _, row in profile_df.iterrows():
        c_id = int(row["customer_id"])
        tenure = float(row["customer_tenure"])
        products = int(row["number_of_active_products"])
        segment_factor = segment_multiplier[row["customer_segment"]]
        age_factor = age_multiplier[row["age_band"]]
        account_factor = account_multiplier[row["primary_account_type"]]
        relationship_factor = 1.15 if row["relationship_status"] == "Primary" else 0.9

        base_balance = 25000 + 3000 * tenure + 4200 * products + 18000 * age_factor + 14000 * account_factor
        base_balance *= segment_factor * relationship_factor
        balance_trend = rng.normal(0.015, 0.05)

        for month in range(1, months + 1):
            month_effect = 0.04 * np.sin((month / 12) * 2 * np.pi + c_id / 35)
            seasonal_jitter = rng.normal(0.0, 0.08)
            balance = base_balance * (1 + balance_trend * month + month_effect + seasonal_jitter)
            balance = max(balance, 5000)

            salary_regular = np.clip(0.78 + 0.12 * age_factor + 0.08 * relationship_factor + rng.normal(0, 0.1), 0.2, 1.0)
            loan_status = np.clip(rng.normal(0.12 + 0.04 * (1 / (1 + tenure / 10)) - 0.04 * segment_factor, 0.15), -0.3, 0.8)
            overdraft = int(rng.binomial(1, max(0.02 + 0.05 * (1 - salary_regular) + 0.08 * max(0, loan_status), 0.0)))

            transaction_count = max(5, int(np.clip(38 + 12 * age_factor + 17 * products + rng.normal(0, 14), 5, 180)))
            avg_transaction_value = max(30, float(np.clip(120 + 70 * account_factor + 25 * products + rng.normal(0, 55), 20, 500)))
            transaction_value_trend = rng.normal(0.02 + 0.05 * (1 - salary_regular), 0.12)
            inflow_outflow_ratio = np.clip(1.08 + 0.18 * salary_regular - 0.22 * max(0, loan_status) + rng.normal(0, 0.12), 0.35, 2.5)
            recurring_payment_activity = rng.beta(2.5, 4.0)
            failed_declined_rate = np.clip(0.02 + 0.06 * (1 - salary_regular) + 0.07 * max(0, loan_status) + rng.normal(0, 0.04), 0.0, 0.5)

            digital_activity = max(3, np.clip(48 + 10 * age_factor + 12 * segment_factor + 8 * products + rng.normal(0, 12), 3, 180))
            digital_change = rng.normal(-0.08 + 0.04 * (1 - salary_regular), 0.18)
            active_channels = int(np.clip(1 + int(np.round(products / 2)) + rng.integers(0, 3), 1, 6))
            
            # FIXED: Reduced multiplier from 0.4 to 0.02 so branch interactions don't instantly bottom out to 0
            branch_interactions = max(0, int(np.clip(5 + rng.normal(0, 1.5) - 0.02 * digital_activity, 0, 12)))
            usage_reduction = float(np.clip(0.03 + max(0, -digital_change) * 0.9 + rng.normal(0, 0.12), 0.0, 0.9))

            complaint_occurrence = int(rng.binomial(1, min(0.25 + 0.12 * max(0, loan_status) + 0.2 * failed_declined_rate + 0.07 * max(0, usage_reduction), 0.8)))
            complaint_resolution_time = float(np.clip(2 + 8 * complaint_occurrence + rng.normal(0, 2), 0, 25))
            first_contact_resolution = np.clip(0.9 - 0.1 * complaint_occurrence + rng.normal(0, 0.1), 0.1, 1.0)
            service_interaction_frequency = max(0, int(np.clip(2 + branch_interactions + complaint_occurrence * 2 + rng.normal(0, 2), 0, 20)))
            failed_service_incidents = int(rng.binomial(1, min(0.05 + 0.12 * complaint_occurrence + 0.13 * failed_declined_rate, 0.6)))
            service_wait_time = float(np.clip(2 + 3 * complaint_occurrence + 5 * failed_service_incidents + rng.normal(0, 3), 0, 18))

            last_meaningful_interaction_days = max(0, int(np.clip(11 + 16 * usage_reduction + 17 * (1 - salary_regular) + rng.normal(0, 6), 2, 90)))

            rows.append(
                {
                    "customer_id": c_id,
                    "month_index": month,
                    "average_account_balance": round(balance, 2),
                    "balance_trend": round(balance_trend + 0.02 * month_effect, 4),
                    "income_salary_inflow_regularity": round(salary_regular, 4),
                    "loan_repayment_status": round(loan_status, 4),
                    "credit_utilization": round(np.clip(0.32 + 0.15 * max(0, loan_status) + 0.12 * (1 - salary_regular) + rng.normal(0, 0.08), 0.0, 1.5), 4),
                    "recent_overdraft_occurrences": overdraft,
                    "transaction_count": int(transaction_count),
                    "average_transaction_value": round(avg_transaction_value, 2),
                    "transaction_value_trend": round(transaction_value_trend, 4),
                    "inflow_to_outflow_ratio": round(inflow_outflow_ratio, 4),
                    "recurring_payment_activity": round(recurring_payment_activity, 4),
                    "failed_declined_transaction_rate": round(failed_declined_rate, 4),
                    "days_since_last_meaningful_interaction": last_meaningful_interaction_days,
                    "digital_banking_activity_level": round(digital_activity, 2),
                    "change_in_digital_activity": round(digital_change, 4),
                    "number_of_active_banking_channels": active_channels,
                    "branch_service_interaction_frequency": branch_interactions,
                    "unusual_reduction_in_account_usage": round(usage_reduction, 4),
                    "complaint_occurrence": complaint_occurrence,
                    "complaint_resolution_time": round(complaint_resolution_time, 2),
                    "first_contact_resolution": round(first_contact_resolution, 4),
                    "service_interaction_frequency": service_interaction_frequency,
                    "failed_service_transaction_incidents": failed_service_incidents,
                    "service_channel_wait_time": round(service_wait_time, 2),
                }
            )

    return pd.DataFrame(rows)
