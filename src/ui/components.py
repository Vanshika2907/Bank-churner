from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import streamlit as st


def render_home() -> None:
    st.title("Bank Customer Relationship Exit Prediction")
    st.markdown(
        """
        Estimate a customer's likelihood of exiting their banking relationship using behavioral, financial, engagement, and service indicators.
        """
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Prediction Horizon", "6 Months")
    col2.metric("Model", "Logistic Regression")
    col3.metric("Data Status", "Synthetic / Academic")

    st.subheader("Overview")
    st.info(
        "This decision-support prototype predicts relationship-exit risk using information available at the prediction date and is intended for academic experimentation."
    )

    st.markdown(
        """
        ### Project scope
        - Customer relationship context
        - Financial health and balance behavior
        - Transaction patterns
        - Engagement and channel usage
        - Service friction and complaint history
        """
    )


def risk_badge(probability: float) -> str:
    if probability < 0.30:
        return "Low"
    if probability < 0.60:
        return "Moderate"
    if probability < 0.80:
        return "High"
    return "Critical"


def render_prediction_result(result: Dict[str, Any]) -> None:
    probability = float(result["probability"])
    pct = float(result["probability_percent"])
    risk = result["risk_category"]
    label = result["label"]

    st.subheader("Relationship Exit Risk")
    st.metric("Probability", f"{pct:.2f}%")
    st.progress(probability)

    if risk == "Low":
        color = "#22c55e"
    elif risk == "Moderate":
        color = "#f59e0b"
    elif risk == "High":
        color = "#f97316"
    else:
        color = "#ef4444"

    st.markdown(
        f"<div style='padding: 16px; border-radius: 12px; background: {color}22; border: 1px solid {color};'>"
        f"<strong>Risk:</strong> {risk}<br>"
        f"<strong>Prediction:</strong> {label}"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_explainability_panel(result: Dict[str, Any]) -> None:
    st.subheader("Why this prediction?")

    top_features = result.get("top_features", [])
    if not top_features:
        st.warning("Model explanations are not currently available for this artifact.")
        return

    feature_names = [item["name"] for item in top_features]
    importance_values = [item["importance"] for item in top_features]
    importance_df = pd.DataFrame({"Importance": importance_values}, index=feature_names)
    st.bar_chart(importance_df)

    top_names = ", ".join(feature_names[:3])
    st.write(
        f"The customer's risk is elevated mainly because {top_names} are among the strongest contributors to exit propensity in the model."
    )


def render_model_diagnostics() -> None:
    from src.model.predictor import get_model_usage_audit, load_predictor, FEATURE_ORDER

    audit = get_model_usage_audit()
    st.title("Model Diagnostics")
    st.caption("Developer-only audit of the actual training and inference pipeline.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Model Type", audit["model_type"])
    col2.metric("Input Features Used", len(audit["model_input_features"]))
    col3.metric("Intended Features", audit["intended_feature_count"])

    st.subheader("Model Information")
    st.code(f"Model file: {audit['model_file']}\nPreprocessor: {audit['preprocessor_type']}\nFeatures used by model: {audit['model_input_features']}")

    st.subheader("Feature Connection Status")
    usage_df = pd.DataFrame(audit["feature_usage_report"])
    st.dataframe(usage_df, use_container_width=True)

    st.metric("Missing Features", len(audit["missing_features"]))
    if audit["missing_features"]:
        st.write("Missing:", audit["missing_features"])
    else:
        st.write("No missing features detected in the current model schema.")

    st.subheader("Prediction Sensitivity Test")
    baseline_payload = {
        "age_band": "46-55",
        "customer_tenure": 8.5,
        "customer_segment": "Mass Affluent",
        "number_of_active_products": 3,
        "primary_account_type": "Current",
        "relationship_status": "Primary",
        "average_account_balance": 120000,
        "balance_trend": 0.01,
        "income_salary_inflow_regularity": 0.92,
        "loan_repayment_status": 0.15,
        "credit_utilization": 0.35,
        "recent_overdraft_occurrences": 1,
        "transaction_count": 120,
        "average_transaction_value": 220,
        "transaction_value_trend": -0.08,
        "inflow_to_outflow_ratio": 1.18,
        "recurring_payment_activity": 0.62,
        "failed_declined_transaction_rate": 0.04,
        "days_since_last_meaningful_interaction": 18,
        "digital_banking_activity_level": 85,
        "change_in_digital_activity": -0.2,
        "number_of_active_banking_channels": 4,
        "branch_service_interaction_frequency": 2,
        "unusual_reduction_in_account_usage": 0.18,
        "complaint_occurrence": 1,
        "complaint_resolution_time": 6,
        "first_contact_resolution": 0.82,
        "service_interaction_frequency": 5,
        "failed_service_transaction_incidents": 2,
        "service_channel_wait_time": 10,
    }
    model = load_predictor()
    baseline_result = model.predict_proba(pd.DataFrame([baseline_payload]))[0, 1]
    st.write(f"Baseline exit probability: {baseline_result * 100:.2f}%")

    selected_feature = st.selectbox("Feature to test", FEATURE_ORDER)
    feature_value = baseline_payload[selected_feature]
    altered_value = st.text_input("Modified value", value=str(feature_value))

    if st.button("Recalculate Prediction"):
        try:
            payload = baseline_payload.copy()
            payload[selected_feature] = type(feature_value)(altered_value) if feature_value is not None else altered_value
            new_result = model.predict_proba(pd.DataFrame([payload]))[0, 1]
            st.success(f"Original: {baseline_result * 100:.2f}% | New: {new_result * 100:.2f}% | Change: {(new_result - baseline_result) * 100:.2f} points")
        except Exception as exc:
            st.error(f"Input conversion error: {exc}")


def render_prediction_simulator() -> None:
    from src.model.predictor import get_prediction_sensitivity_table, load_predictor, FEATURE_ORDER, prepare_input_frame
    import pandas as pd

    st.title("Prediction Simulator")
    st.caption("Developer test page to verify whether model output responds to feature changes.")

    default_payload = {
        "age_band": "46-55",
        "customer_tenure": 8.5,
        "customer_segment": "Mass Affluent",
        "number_of_active_products": 3,
        "primary_account_type": "Current",
        "relationship_status": "Primary",
        "average_account_balance": 120000,
        "balance_trend": 0.01,
        "income_salary_inflow_regularity": 0.92,
        "loan_repayment_status": 0.15,
        "credit_utilization": 0.35,
        "recent_overdraft_occurrences": 1,
        "transaction_count": 120,
        "average_transaction_value": 220,
        "transaction_value_trend": -0.08,
        "inflow_to_outflow_ratio": 1.18,
        "recurring_payment_activity": 0.62,
        "failed_declined_transaction_rate": 0.04,
        "days_since_last_meaningful_interaction": 18,
        "digital_banking_activity_level": 85,
        "change_in_digital_activity": -0.2,
        "number_of_active_banking_channels": 4,
        "branch_service_interaction_frequency": 2,
        "unusual_reduction_in_account_usage": 0.18,
        "complaint_occurrence": 1,
        "complaint_resolution_time": 6,
        "first_contact_resolution": 0.82,
        "service_interaction_frequency": 5,
        "failed_service_transaction_incidents": 2,
        "service_channel_wait_time": 10,
    }

    payload = default_payload.copy()
    selected_feature = st.selectbox("Feature to modify", FEATURE_ORDER)
    value_input = st.text_input("New value", value=str(payload[selected_feature]))

    if st.button("Recalculate Prediction"):
        try:
            payload[selected_feature] = type(default_payload[selected_feature])(value_input)
            model = load_predictor()
            original = model.predict_proba(pd.DataFrame([default_payload]))[0, 1]
            new = model.predict_proba(pd.DataFrame([payload]))[0, 1]
            st.write(f"Original: {original * 100:.2f}%")
            st.write(f"New: {new * 100:.2f}%")
            st.write(f"Difference: {(new - original) * 100:.2f} percentage points")
            st.write("Changed feature:", selected_feature)
        except Exception as exc:
            st.error(f"Validation error: {exc}")

    sensitivity_df = pd.DataFrame(get_prediction_sensitivity_table(default_payload))
    st.dataframe(sensitivity_df, use_container_width=True)
