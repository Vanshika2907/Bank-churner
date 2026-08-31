from __future__ import annotations

from typing import Any, Dict


def validate_customer_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    errors: list[str] = []

    required_fields = [
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

    for field in required_fields:
        if field not in payload or payload[field] is None:
            errors.append(f"Missing required value: {field}")

    if payload.get("customer_tenure", 0) < 0:
        errors.append("Customer tenure cannot be negative.")
    if payload.get("average_account_balance", 0) < 0:
        errors.append("Average account balance cannot be negative.")
    if payload.get("number_of_active_products", 0) < 1:
        errors.append("Number of active products must be at least 1.")
    if payload.get("transaction_count", 0) < 0:
        errors.append("Transaction count cannot be negative.")
    if payload.get("complaint_resolution_time", 0) < 0:
        errors.append("Complaint resolution time cannot be negative.")
    if payload.get("service_channel_wait_time", 0) < 0:
        errors.append("Service channel wait time cannot be negative.")

    return {"valid": len(errors) == 0, "errors": errors}
