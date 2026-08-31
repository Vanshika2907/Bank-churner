from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config_loader import project_root
from src.decision_support.risk_scoring import assign_risk_category, build_priority_table
from src.etl.feature_pipeline import (
    add_engineered_features,
    leakage_guard_checks,
    load_processed_dataset,
    prepare_model_frame,
    run_data_quality_checks,
)
from src.evaluation.model_report import save_experiment_results
from src.models.baseline_models import train_baseline_models

ROOT = project_root()


def main() -> None:
    df = load_processed_dataset(ROOT / "data" / "processed" / "ml_ready_dataset.csv")
    assert df.shape[0] > 0, "Dataset is empty"

    quality = run_data_quality_checks(df)
    print("Data quality checks:")
    print(json.dumps(quality, indent=2, default=str))

    engineered = add_engineered_features(df)
    print("Leakage guard check passed:", leakage_guard_checks(engineered))

    X, y, feature_names = prepare_model_frame(engineered)
    baseline_results = train_baseline_models(X, y, random_state=42)
    save_experiment_results(baseline_results, "baseline_model_results.json")

    model_name = max(baseline_results, key=lambda name: baseline_results[name]["f1"])
    best_model = model_name
    print(f"Best F1 model: {best_model}")

    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    test_probs = model.predict_proba(X_test)[:, 1]
    test_pred = model.predict(X_test)

    decision_df = pd.DataFrame({
        "customer_id": X_test.index.map(lambda idx: df.loc[idx, "customer_id"] if idx in df.index else idx),
        "predicted_exit_probability": test_probs,
        "predicted_label": test_pred,
    })

    decision_df["risk_category"] = assign_risk_category(decision_df["predicted_exit_probability"])
    priority_table = build_priority_table(decision_df, probability_col="predicted_exit_probability")
    priority_table.to_csv(ROOT / "reports" / "results" / "priority_table.csv", index=False)

    print(priority_table.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
