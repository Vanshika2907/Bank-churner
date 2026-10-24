from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import streamlit as st

from src.model.predictor import (
    FEATURE_GROUPS,
    get_available_models,
    load_predictor,
    predict_customer,
)
from src.ui.components import render_explainability_panel, render_home, render_model_diagnostics, render_prediction_result
from src.utils.validation import validate_customer_payload

st.set_page_config(
    page_title="Bank Customer Relationship Exit Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

SPECS_PATH = Path(__file__).parent / "config" / "feature_specs.json"

CONSTANT_FIELDS = {"branch_service_interaction_frequency": 0}

SAMPLE_PROFILES: Dict[str, Dict[str, Any]] = {
    "Low-Risk Customer (Loyal)": {
        "age_band": "46-55",
        "customer_tenure": 12.0,
        "customer_segment": "Mass Affluent",
        "number_of_active_products": 4,
        "primary_account_type": "Current",
        "relationship_status": "Primary",
        "average_account_balance": 185000.0,
        "balance_trend": 0.08,
        "income_salary_inflow_regularity": 0.98,
        "loan_repayment_status": 0.0,
        "credit_utilization": 0.22,
        "recent_overdraft_occurrences": 0,
        "transaction_count": 145,
        "average_transaction_value": 310.0,
        "transaction_value_trend": 0.05,
        "inflow_to_outflow_ratio": 1.28,
        "recurring_payment_activity": 0.85,
        "failed_declined_transaction_rate": 0.01,
        "days_since_last_meaningful_interaction": 8,
        "digital_banking_activity_level": 92.0,
        "change_in_digital_activity": 0.15,
        "number_of_active_banking_channels": 4,
        "branch_service_interaction_frequency": 0,
        "unusual_reduction_in_account_usage": 0.02,
        "complaint_occurrence": 0,
        "complaint_resolution_time": 0,
        "first_contact_resolution": 0.95,
        "service_interaction_frequency": 3,
        "failed_service_transaction_incidents": 0,
        "service_channel_wait_time": 4,
    },
    "High-Risk Customer (Churn)": {
        "age_band": "26-35",
        "customer_tenure": 1.5,
        "customer_segment": "Mass",
        "number_of_active_products": 1,
        "primary_account_type": "Current",
        "relationship_status": "Secondary",
        "average_account_balance": 12500.0,
        "balance_trend": -0.45,
        "income_salary_inflow_regularity": 0.35,
        "loan_repayment_status": 0.80,
        "credit_utilization": 0.88,
        "recent_overdraft_occurrences": 4,
        "transaction_count": 28,
        "average_transaction_value": 65.0,
        "transaction_value_trend": -0.38,
        "inflow_to_outflow_ratio": 1.09,
        "recurring_payment_activity": 0.20,
        "failed_declined_transaction_rate": 0.12,
        "days_since_last_meaningful_interaction": 65,
        "digital_banking_activity_level": 25.0,
        "change_in_digital_activity": -0.65,
        "number_of_active_banking_channels": 1,
        "branch_service_interaction_frequency": 0,
        "unusual_reduction_in_account_usage": 0.72,
        "complaint_occurrence": 1,
        "complaint_resolution_time": 18,
        "first_contact_resolution": 0.75,
        "service_interaction_frequency": 8,
        "failed_service_transaction_incidents": 4,
        "service_channel_wait_time": 22,
    },
    "Moderate-Risk Customer": {
        "age_band": "36-45",
        "customer_tenure": 5.0,
        "customer_segment": "Affluent",
        "number_of_active_products": 2,
        "primary_account_type": "Savings",
        "relationship_status": "Secondary",
        "average_account_balance": 65000.0,
        "balance_trend": -0.10,
        "income_salary_inflow_regularity": 0.70,
        "loan_repayment_status": 0.10,
        "credit_utilization": 0.50,
        "recent_overdraft_occurrences": 1,
        "transaction_count": 60,
        "average_transaction_value": 150.0,
        "transaction_value_trend": -0.12,
        "inflow_to_outflow_ratio": 1.15,
        "recurring_payment_activity": 0.50,
        "failed_declined_transaction_rate": 0.04,
        "days_since_last_meaningful_interaction": 35,
        "digital_banking_activity_level": 55.0,
        "change_in_digital_activity": -0.25,
        "number_of_active_banking_channels": 2,
        "branch_service_interaction_frequency": 0,
        "unusual_reduction_in_account_usage": 0.30,
        "complaint_occurrence": 1,
        "complaint_resolution_time": 8,
        "first_contact_resolution": 0.85,
        "service_interaction_frequency": 5,
        "failed_service_transaction_incidents": 1,
        "service_channel_wait_time": 12,
    },
}


def load_specs() -> Dict[str, dict]:
    if not SPECS_PATH.exists():
        from build_feature_specs import build_specs

        build_specs()
    with open(SPECS_PATH) as f:
        return json.load(f)


def get_default_payload(specs: Dict[str, dict]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for field, spec in specs.items():
        if spec.get("type") == "categorical":
            payload[field] = spec["options"][0]
        else:
            payload[field] = spec.get("default", spec.get("min", 0))
    for field, val in CONSTANT_FIELDS.items():
        payload[field] = val
    return payload


def load_sample_profile(profile_data: Dict[str, Any]) -> None:
    st.session_state.customer_payload = profile_data.copy()
    for k, v in profile_data.items():
        st.session_state[f"input_{k}"] = v


def render_field(field: str, specs: Dict[str, dict], saved: object = None) -> object:
    label = field.replace("_", " ").title()
    widget_key = f"input_{field}"

    if field in CONSTANT_FIELDS:
        st.number_input(
            label,
            value=int(CONSTANT_FIELDS[field]),
            disabled=True,
            key=widget_key,
            help="Known limitation: this feature is constant (0) in the training data, so the model cannot use it.",
        )
        return CONSTANT_FIELDS[field]

    spec = specs.get(field)
    if spec is None:
        st.error(f"No spec found for '{field}'. Re-run build_feature_specs.py.")
        st.stop()

    if spec["type"] == "categorical":
        options = spec["options"]
        saved_val = st.session_state.get(widget_key, saved)
        index = options.index(saved_val) if saved_val in options else 0
        return st.selectbox(label, options, index=index, key=widget_key)

    if spec["type"] == "int":
        lo, hi, step = int(spec["min"]), int(spec["max"]), int(spec["step"])
        saved_val = st.session_state.get(widget_key, saved)
        value = int(saved_val) if saved_val is not None else int(spec.get("default", lo))
        value = min(max(value, lo), hi)
        return st.number_input(label, min_value=lo, max_value=hi, value=value, step=step, key=widget_key)

    lo, hi, step = float(spec["min"]), float(spec["max"]), float(spec["step"])
    saved_val = st.session_state.get(widget_key, saved)
    value = float(saved_val) if saved_val is not None else float(spec.get("default", lo))
    value = min(max(value, lo), hi)
    return st.number_input(label, min_value=lo, max_value=hi, value=value, step=step, key=widget_key)


def render_home_page() -> None:
    render_home()


def render_prediction_page(model_name: str = None) -> None:
    st.header("Customer Exit Prediction")
    st.caption("The prediction is generated by the saved multi-model pipeline; the form collects the raw inputs used by the pipeline.")

    specs = load_specs()

    # Pre-populate session state with valid defaults to prevent 'Missing required value' errors
    if "customer_payload" not in st.session_state or not st.session_state.customer_payload:
        st.session_state.customer_payload = get_default_payload(specs)

    # 1. Preset Profile Toolbar for Lab Evaluations
    st.markdown("### 🧪 Lab Quick-Test: Preset Customer Profiles")
    st.caption("1-click loads all 30 parameters instantly across all steps, allowing immediate testing.")
    col_s1, col_s2, col_s3, col_s4 = st.columns([1, 1, 1, 1.2])

    if col_s1.button("🟢 Low-Risk Sample", use_container_width=True):
        load_sample_profile(SAMPLE_PROFILES["Low-Risk Customer (Loyal)"])
        st.session_state.sample_loaded_name = "Low-Risk Customer (Loyal)"
        st.session_state.last_prediction = None
        st.rerun()

    if col_s2.button("🔴 High-Risk Sample", use_container_width=True):
        load_sample_profile(SAMPLE_PROFILES["High-Risk Customer (Churn)"])
        st.session_state.sample_loaded_name = "High-Risk Customer (Churn)"
        st.session_state.last_prediction = None
        st.rerun()

    if col_s3.button("🟡 Moderate-Risk", use_container_width=True):
        load_sample_profile(SAMPLE_PROFILES["Moderate-Risk Customer"])
        st.session_state.sample_loaded_name = "Moderate-Risk Customer"
        st.session_state.last_prediction = None
        st.rerun()

    if col_s4.button("⚡ Instant Predict All", use_container_width=True):
        validation = validate_customer_payload(st.session_state.customer_payload)
        if not validation["valid"]:
            for message in validation["errors"]:
                st.error(message)
        else:
            predictor = load_predictor(model_name)
            result = predict_customer(st.session_state.customer_payload, predictor, model_name=model_name)
            st.session_state.last_prediction = result

    if st.session_state.get("sample_loaded_name"):
        st.success(f"✓ Loaded profile: **{st.session_state.sample_loaded_name}** (All 30 fields populated across Steps 1–5)")

    # Display prediction result banner if available
    if st.session_state.get("last_prediction"):
        st.markdown("---")
        render_prediction_result(st.session_state.last_prediction)
        render_explainability_panel(st.session_state.last_prediction)
        st.markdown("---")

    # 2. Existing 5-Step Guided Form Wizard
    step_order = list(FEATURE_GROUPS.keys())
    if "prediction_step_index" not in st.session_state:
        st.session_state.prediction_step_index = 0

    current_step = min(max(0, st.session_state.prediction_step_index), len(step_order) - 1)
    step_name = step_order[current_step]
    step_fields = FEATURE_GROUPS[step_name]

    st.markdown(f"#### Step {current_step + 1} of {len(step_order)}: {step_name}")
    st.progress((current_step + 1) / len(step_order))

    payload = st.session_state.customer_payload.copy()
    for field in step_fields:
        payload[field] = render_field(field, specs, payload.get(field))

    st.session_state.customer_payload = payload

    col_prev, col_next, col_submit = st.columns([1, 1, 2])
    if current_step > 0:
        if col_prev.button("⬅ Back"):
            st.session_state.prediction_step_index = max(0, current_step - 1)
            st.rerun()

    if current_step < len(step_order) - 1:
        if col_next.button("Next ➡"):
            st.session_state.prediction_step_index = current_step + 1
            st.rerun()
    else:
        if col_submit.button("🎯 Predict Relationship Exit", use_container_width=True):
            validation = validate_customer_payload(payload)
            if not validation["valid"]:
                for message in validation["errors"]:
                    st.error(message)
                return

            predictor = load_predictor(model_name)
            result = predict_customer(payload, predictor, model_name=model_name)
            st.session_state.last_prediction = result
            st.session_state.customer_payload = payload
            st.rerun()


def main() -> None:
    st.sidebar.title("Project Overview")
    st.sidebar.caption("University ML Lab Project")
    st.sidebar.markdown("**Project:** Bank Customer Relationship Exit Prediction")
    st.sidebar.markdown("**Focus:** 6-month relationship exit risk")
    st.sidebar.markdown("**Data status:** Synthetic, academic use only")

    models = get_available_models()
    model_labels = [
        f"{m['name']} ({m['role']})" + (" ★ Best Model" if m.get("is_best") else "")
        for m in models
    ]
    model_map = {label: m["key"] for label, m in zip(model_labels, models)}

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Pipeline Classifier:**")
    selected_label = st.sidebar.selectbox("Active Model", model_labels)
    selected_key = model_map[selected_label]

    st.sidebar.markdown("---")
    project_tab = st.sidebar.radio("Navigate", ["Home", "Prediction", "Diagnostics"])

    if project_tab == "Home":
        render_home_page()
    elif project_tab == "Prediction":
        render_prediction_page(model_name=selected_key)
    else:
        render_model_diagnostics()


if __name__ == "__main__":
    main()
