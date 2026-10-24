from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config_loader import project_root
from src.decision_support.risk_scoring import assign_risk_category, build_priority_table
from src.etl.feature_pipeline import (
    leakage_guard_checks,
    load_processed_dataset,
    run_data_quality_checks,
)
from src.models.churn_models import train_and_evaluate_all_models, BUNDLE_PATH, PRIMARY_MODEL_PATH

ROOT = project_root()


def main() -> None:
    print("=" * 80)
    print("BANK CHURN PREDICTION PIPELINE - MODEL EXECUTION & EXPORT")
    print("=" * 80)

    dataset_path = ROOT / "data" / "processed" / "ml_ready_dataset.csv"
    df = load_processed_dataset(dataset_path)
    assert df.shape[0] > 0, "Dataset is empty"

    # 1. Data Quality Checks
    quality = run_data_quality_checks(df)
    print("\n[Step 1] Data Quality Checks:")
    print(json.dumps(quality, indent=2, default=str))

    # 2. Leakage Guard
    leakage_passed = leakage_guard_checks(df)
    print(f"\n[Step 2] Leakage Guard Check Passed: {leakage_passed}")

    # 3. Model Training, Unsupervised Feature Engineering & Metric Calculation
    print("\n[Step 3] Training ML Models with Unsupervised Feature Engineers...")
    bundle = train_and_evaluate_all_models(df, random_state=42)
    metrics = bundle["metrics"]

    # 4. Summary Table of Result Metrics
    print("\n" + "=" * 96)
    print("MODEL PERFORMANCE & METRIC SUMMARY")
    print("=" * 96)
    header = f"{'Model':<22} | {'Pipeline Role':<24} | {'ROC-AUC':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'Confusion Matrix (TN,FP,FN,TP)'}"
    print(header)
    print("-" * 96)

    for key, res in metrics.items():
        cm = res["confusion_matrix"]
        cm_str = f"[{cm['tn']}, {cm['fp']}, {cm['fn']}, {cm['tp']}]"
        print(
            f"{res['display_name']:<22} | {res['role']:<24} | {res['roc_auc']:<8.4f} | {res['precision']:<9.4f} | {res['recall']:<8.4f} | {res['f1']:<8.4f} | {cm_str}"
        )
    print("-" * 96)

    # 5. Feature Importances Summary
    print("\n" + "=" * 96)
    print("TOP PREDICTIVE FEATURE IMPORTANCES BY MODEL")
    print("=" * 96)
    for key, res in metrics.items():
        print(f"\n▶ {res['display_name']} ({res['role']}):")
        for i, item in enumerate(res["top_features"][:5], 1):
            dir_str = f" ({item.get('direction', '')})" if "direction" in item else ""
            print(f"   {i}. {item['feature']:<38} (importance: {item['importance']:.4f}){dir_str}")

    # 6. Customer Prioritization Table with Best Model
    best_model_name = bundle["best_model_name"]
    best_pipeline = bundle["best_pipeline"]
    print(f"\n[Step 4] Generating Priority Table using Best Model ({metrics[best_model_name]['display_name']})...")

    X_raw = df.drop(columns=["customer_id", "relationship_exit"], errors="ignore")
    y = df["relationship_exit"]
    _, X_test, _, y_test = train_test_split(X_raw, y, test_size=0.2, stratify=y, random_state=42)

    test_probs = best_pipeline.predict_proba(X_test)[:, 1]
    test_pred = best_pipeline.predict(X_test)

    decision_df = pd.DataFrame({
        "customer_id": X_test.index.map(lambda idx: df.loc[idx, "customer_id"] if idx in df.index else idx),
        "predicted_exit_probability": test_probs,
        "predicted_label": test_pred,
    })

    decision_df["risk_category"] = assign_risk_category(decision_df["predicted_exit_probability"])
    priority_table = build_priority_table(decision_df, probability_col="predicted_exit_probability")
    priority_csv_path = ROOT / "reports" / "results" / "priority_table.csv"
    priority_table.to_csv(priority_csv_path, index=False)

    print(f"Priority table saved to: {priority_csv_path}")
    print("\nTop 10 High-Risk Customers Identified for Intervention:")
    print(priority_table.head(10).to_string(index=False))

    # 7. PKL File Status
    print("\n" + "=" * 80)
    print("SERIALIZATION STATUS:")
    print(f"✓ Unified multi-model .pkl bundle : {BUNDLE_PATH} ({BUNDLE_PATH.stat().st_size / 1024:.1f} KB)")
    print(f"✓ Primary best-model .pkl         : {PRIMARY_MODEL_PATH} ({PRIMARY_MODEL_PATH.stat().st_size / 1024:.1f} KB)")
    print("=" * 80)


if __name__ == "__main__":
    main()
