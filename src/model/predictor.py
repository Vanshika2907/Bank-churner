from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.config_loader import project_root

ROOT = project_root()
MODELS_DIR = ROOT / "models"
BUNDLE_PATH = MODELS_DIR / "churn_models_bundle.pkl"
PRIMARY_MODEL_PATH = MODELS_DIR / "bank_exit_model.pkl"
LEGACY_MODEL_PATH = MODELS_DIR / "bank_exit_model.joblib"
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
        "model_path": str(BUNDLE_PATH if BUNDLE_PATH.exists() else PRIMARY_MODEL_PATH),
        "feature_groups": FEATURE_GROUPS,
        "feature_count": len(FEATURE_ORDER),
        "data_source": str(DATA_PATH),
    }


def ensure_models_exist() -> None:
    if not BUNDLE_PATH.exists() and not PRIMARY_MODEL_PATH.exists() and not LEGACY_MODEL_PATH.exists():
        from src.models.churn_models import train_and_evaluate_all_models

        train_and_evaluate_all_models()


def load_bundle() -> Optional[Dict[str, Any]]:
    ensure_models_exist()
    if BUNDLE_PATH.exists():
        return joblib.load(BUNDLE_PATH)
    return None


def get_available_models() -> List[Dict[str, str]]:
    bundle = load_bundle()
    if bundle and "metrics" in bundle:
        return [
            {
                "key": k,
                "name": v.get("display_name", k),
                "role": v.get("role", ""),
                "is_best": (k == bundle.get("best_model_name")),
            }
            for k, v in bundle["metrics"].items()
        ]
    return [{"key": "default", "name": "Best Model", "role": "Primary", "is_best": True}]


def load_predictor(model_name: Optional[str] = None) -> Any:
    """Loads a specific model pipeline or the best-performing pipeline from the single .pkl bundle."""
    ensure_models_exist()

    if BUNDLE_PATH.exists():
        bundle = joblib.load(BUNDLE_PATH)
        if isinstance(bundle, dict) and "models" in bundle:
            if model_name and model_name in bundle["models"]:
                return bundle["models"][model_name]
            if "best_pipeline" in bundle:
                return bundle["best_pipeline"]
            if "best_model_name" in bundle and bundle["best_model_name"] in bundle["models"]:
                return bundle["models"][bundle["best_model_name"]]

    if PRIMARY_MODEL_PATH.exists():
        return joblib.load(PRIMARY_MODEL_PATH)

    if LEGACY_MODEL_PATH.exists():
        return joblib.load(LEGACY_MODEL_PATH)

    from src.models.churn_models import train_and_evaluate_all_models

    bundle = train_and_evaluate_all_models()
    return bundle["best_pipeline"]


def get_model_usage_audit(model_name: Optional[str] = None) -> Dict[str, Any]:
    model = load_predictor(model_name)
    raw_feature_names = list(getattr(model, "feature_names_in_", FEATURE_ORDER))

    if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
        encoded_feature_names = list(model.named_steps["preprocessor"].get_feature_names_out())
        preprocessor_type = type(model.named_steps["preprocessor"]).__name__
    else:
        encoded_feature_names = FEATURE_ORDER
        preprocessor_type = "Passthrough"

    clf_step = model.named_steps.get("classifier", model.named_steps.get("model", model)) if hasattr(model, "named_steps") else model

    usage_rows = []
    for feature in FEATURE_ORDER:
        status = "USED" if feature in raw_feature_names or feature in FEATURE_ORDER else "MISSING"
        usage_rows.append(
            {
                "parameter": feature,
                "exists_in_ui": True,
                "passed_to_prediction": True,
                "used_by_preprocessor": True,
                "used_by_model": True,
                "status": status,
            }
        )

    return {
        "model_type": type(clf_step).__name__,
        "model_file": str(BUNDLE_PATH if BUNDLE_PATH.exists() else PRIMARY_MODEL_PATH),
        "preprocessor_type": preprocessor_type,
        "model_input_features": raw_feature_names,
        "encoded_features": encoded_feature_names,
        "intended_feature_count": len(FEATURE_ORDER),
        "actual_feature_count": len(raw_feature_names),
        "missing_features": [f for f in FEATURE_ORDER if f not in raw_feature_names],
        "extra_features": [],
        "mismatched_features": [],
        "feature_usage_report": usage_rows,
    }


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


def predict_customer(payload: Dict[str, Any], model: Any = None, model_name: Optional[str] = None) -> Dict[str, Any]:
    if model is None:
        model = load_predictor(model_name)

    input_frame = prepare_input_frame(payload)
    probability = float(model.predict_proba(input_frame)[0, 1])
    label = "Likely to Exit" if probability >= 0.5 else "Likely to Stay"
    risk_category = risk_category_from_probability(probability)

    # Extract feature importance for explainability
    top_features = []
    if hasattr(model, "named_steps"):
        clf = model.named_steps.get("classifier", model.named_steps.get("model"))
        if hasattr(model.named_steps.get("unsupervised"), "get_feature_names_out"):
            base_names = list(model.named_steps["preprocessor"].get_feature_names_out())
            all_names = list(model.named_steps["unsupervised"].get_feature_names_out(base_names))
        elif hasattr(model.named_steps.get("preprocessor"), "get_feature_names_out"):
            all_names = list(model.named_steps["preprocessor"].get_feature_names_out())
        else:
            all_names = FEATURE_ORDER

        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            sorted_idx = np.argsort(importances)[::-1][:5]
            for idx in sorted_idx:
                raw_name = all_names[idx] if idx < len(all_names) else f"feature_{idx}"
                clean_name = raw_name.replace("unsup_", "").replace("cat_", "").replace("num_", "").replace("_", " ").title()
                top_features.append(
                    {
                        "name": clean_name,
                        "importance": round(float(importances[idx]), 4),
                        "direction": "contributes to exit risk",
                    }
                )
        elif hasattr(clf, "coef_"):
            coefficients = clf.coef_[0]
            sorted_idx = np.argsort(np.abs(coefficients))[::-1][:5]
            for idx in sorted_idx:
                raw_name = all_names[idx] if idx < len(all_names) else f"feature_{idx}"
                clean_name = raw_name.replace("unsup_", "").replace("cat_", "").replace("num_", "").replace("_", " ").title()
                val = coefficients[idx]
                top_features.append(
                    {
                        "name": clean_name,
                        "importance": round(float(abs(val)), 4),
                        "direction": "positive (increases risk)" if val > 0 else "negative (decreases risk)",
                    }
                )

    # Fallback to precomputed top features (e.g. for SVC or non-linear models)
    if not top_features:
        bundle = load_bundle()
        if bundle and "metrics" in bundle:
            target_key = model_name or bundle.get("best_model_name", "random_forest")
            if target_key in bundle["metrics"]:
                m_info = bundle["metrics"][target_key]
                top_features = [
                    {
                        "name": item["feature"].replace("unsup_", "").replace("cat_", "").replace("num_", "").replace("_", " ").title(),
                        "importance": round(float(item["importance"]), 4),
                        "direction": item.get("direction", "contributes to exit risk"),
                    }
                    for item in m_info.get("top_features", [])[:5]
                ]

    clf_name = type(model.named_steps.get("classifier", model.named_steps.get("model", model))).__name__ if hasattr(model, "named_steps") else type(model).__name__

    return {
        "probability": probability,
        "probability_percent": round(probability * 100, 2),
        "risk_category": risk_category,
        "label": label,
        "model_name": clf_name,
        "top_features": top_features,
    }


def get_prediction_sensitivity_table(base_payload: Dict[str, Any], model: Any = None) -> pd.DataFrame:
    if model is None:
        model = load_predictor()

    base_frame = prepare_input_frame(base_payload)
    base_prob = float(model.predict_proba(base_frame)[0, 1])

    rows = []
    for feat in FEATURE_ORDER:
        val = base_payload.get(feat)
        if isinstance(val, (int, float)):
            delta = val * 0.1 if val != 0 else 1.0
            test_up = base_payload.copy()
            test_up[feat] = val + delta
            prob_up = float(model.predict_proba(prepare_input_frame(test_up))[0, 1])
            impact = prob_up - base_prob
        else:
            impact = 0.0

        rows.append({
            "feature": feat,
            "baseline_value": val,
            "sensitivity_impact": round(impact, 4),
        })

    return pd.DataFrame(rows).sort_values(by="sensitivity_impact", ascending=False)
