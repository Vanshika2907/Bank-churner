from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "bank_exit_model.joblib"
DATA_PATH = ROOT / "data" / "processed" / "ml_ready_dataset.csv"

FEATURE_GROUPS: Dict[str, List[str]] = {
    "A. Customer & Relationship Context": [
        "age_band",
        "customer_tenure",
        "customer_segment",
        "number_of_active_products",
        "primary_account_type",
        "relationship_status",
    ],
    "B. Financial Position": [
        "average_account_balance",
        "balance_trend",
        "income_salary_inflow_regularity",
        "loan_repayment_status",
        "credit_utilization",
        "recent_overdraft_occurrences",
    ],
    "C. Transaction Behavior": [
        "transaction_count",
        "average_transaction_value",
        "transaction_value_trend",
        "inflow_to_outflow_ratio",
        "recurring_payment_activity",
        "failed_declined_transaction_rate",
    ],
    "D. Engagement & Channel Usage": [
        "days_since_last_meaningful_interaction",
        "digital_banking_activity_level",
        "change_in_digital_activity",
        "number_of_active_banking_channels",
        "branch_service_interaction_frequency",
        "unusual_reduction_in_account_usage",
    ],
    "E. Service & Relationship Friction": [
        "complaint_occurrence",
        "complaint_resolution_time",
        "first_contact_resolution",
        "service_interaction_frequency",
        "failed_service_transaction_incidents",
        "service_channel_wait_time",
    ],
}

FEATURE_ORDER = [item for group in FEATURE_GROUPS.values() for item in group]
CATEGORICAL_FEATURES = [
    "age_band",
    "customer_segment",
    "primary_account_type",
    "relationship_status",
]
NUMERIC_FEATURES = [f for f in FEATURE_ORDER if f not in CATEGORICAL_FEATURES]


def get_prediction_context() -> Dict[str, Any]:
    return {
        "model_path": str(MODEL_PATH),
        "feature_groups": FEATURE_GROUPS,
        "feature_count": len(FEATURE_ORDER),
        "data_source": str(DATA_PATH),
    }


def get_model_usage_audit() -> Dict[str, Any]:
    model = load_predictor()
    raw_feature_names = list(getattr(model, "feature_names_in_", FEATURE_ORDER))
    encoded_feature_names = list(model.named_steps["preprocessor"].get_feature_names_out())

    usage_rows = []
    for feature in FEATURE_ORDER:
        status = "USED" if feature in raw_feature_names else "MISSING"
        usage_rows.append(
            {
                "parameter": feature,
                "exists_in_ui": True,
                "passed_to_prediction": feature in raw_feature_names,
                "used_by_preprocessor": feature in raw_feature_names,
                "used_by_model": feature in raw_feature_names,
                "status": status,
            }
        )

    return {
        "model_type": type(model).__name__,
        "model_file": str(MODEL_PATH),
        "preprocessor_type": type(model.named_steps["preprocessor"]).__name__,
        "model_input_features": raw_feature_names,
        "encoded_features": encoded_feature_names,
        "intended_feature_count": len(FEATURE_ORDER),
        "actual_feature_count": len(raw_feature_names),
        "missing_features": [f for f in FEATURE_ORDER if f not in raw_feature_names],
        "extra_features": [],
        "mismatched_features": [],
        "feature_usage_report": usage_rows,
    }


def train_default_model_if_missing() -> None:
    if MODEL_PATH.exists():
        return

    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["customer_id", "relationship_exit"], errors="ignore")
    y = df["relationship_exit"]

    categorical = [col for col in CATEGORICAL_FEATURES if col in X.columns]
    numerical = [col for col in NUMERIC_FEATURES if col in X.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)


def load_predictor() -> Any:
    train_default_model_if_missing()
    return joblib.load(MODEL_PATH)


def risk_category_from_probability(probability: float) -> str:
    if probability < 0.30:
        return "Low"
    if probability < 0.60:
        return "Moderate"
    if probability < 0.80:
        return "High"
    return "Critical"


def prepare_input_frame(payload: Dict[str, Any]) -> pd.DataFrame:
    row = {field: payload.get(field) for field in FEATURE_ORDER}
    return pd.DataFrame([row])


def predict_customer(payload: Dict[str, Any], model: Any) -> Dict[str, Any]:
    input_frame = prepare_input_frame(payload)
    probability = float(model.predict_proba(input_frame)[0, 1])
    label = "Likely to Exit" if probability >= 0.5 else "Likely to Stay"
    risk_category = risk_category_from_probability(probability)

    feature_names = list(model.named_steps["preprocessor"].get_feature_names_out())
    if hasattr(model.named_steps["model"], "coef_"):
        coefficients = model.named_steps["model"].coef_[0]
        feature_importance = sorted(
            zip(feature_names, coefficients),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:5]
    else:
        feature_importance = []

    return {
        "probability": probability,
        "probability_percent": round(probability * 100, 2),
        "risk_category": risk_category,
        "label": label,
        "top_features": [
            {
                "name": name.replace("num__", "").replace("cat__", "").replace("_", " ").title(),
                "importance": round(float(abs(weight)), 4),
                "direction": "positive" if weight > 0 else "negative",
            }
            for name, weight in feature_importance
        ],
    }


def _perturb_feature(feature: str, value: Any) -> Any:
    if feature in CATEGORICAL_FEATURES:
        choices = {
            "age_band": ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
            "customer_segment": ["Mass", "Mass Affluent", "Premium", "SME", "Wealth"],
            "primary_account_type": ["Current", "Savings", "Business", "Private"],
            "relationship_status": ["Primary", "Secondary"],
        }[feature]
        current_index = choices.index(value) if value in choices else 0
        return choices[(current_index + 1) % len(choices)]

    if feature in {"customer_tenure", "average_account_balance", "transaction_count", "average_transaction_value", "days_since_last_meaningful_interaction", "digital_banking_activity_level", "number_of_active_banking_channels", "branch_service_interaction_frequency", "service_interaction_frequency", "complaint_resolution_time", "service_channel_wait_time"}:
        return float(value) * 0.5 if value else 0.0

    if feature in {"balance_trend", "transaction_value_trend", "change_in_digital_activity", "income_salary_inflow_regularity", "loan_repayment_status", "credit_utilization", "inflow_to_outflow_ratio", "recurring_payment_activity", "failed_declined_transaction_rate", "first_contact_resolution", "unusual_reduction_in_account_usage"}:
        return float(value) * 0.7 if value else 0.0

    if feature in {"recent_overdraft_occurrences", "complaint_occurrence", "failed_service_transaction_incidents"}:
        return max(0, int(value) + 1)

    return value


def get_prediction_sensitivity_table(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    model = load_predictor()
    baseline = predict_customer(payload, model)
    baseline_prob = baseline["probability"]
    rows: List[Dict[str, Any]] = []

    for feature in FEATURE_ORDER:
        modified = dict(payload)
        original_value = payload.get(feature)
        modified[feature] = _perturb_feature(feature, original_value)
        new_prediction = predict_customer(modified, model)
        new_prob = new_prediction["probability"]
        delta = new_prob - baseline_prob
        rows.append(
            {
                "feature": feature,
                "baseline": original_value,
                "modified": modified[feature],
                "original_prediction": round(baseline_prob * 100, 2),
                "modified_prediction": round(new_prob * 100, 2),
                "difference_pct_points": round((new_prob - baseline_prob) * 100, 2),
                "changed": abs(delta) > 1e-6,
            }
        )

    return rows
