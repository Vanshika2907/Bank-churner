from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.config_loader import project_root
from src.etl.feature_pipeline import add_engineered_features, load_processed_dataset
from src.models.unsupervised_features import DomainFeatureEngineer, UnsupervisedFeatureEngineer

ROOT = project_root()
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports" / "results"
BUNDLE_PATH = MODELS_DIR / "churn_models_bundle.pkl"
PRIMARY_MODEL_PATH = MODELS_DIR / "bank_exit_model.pkl"

CATEGORICAL_FEATURES = [
    "age_band",
    "customer_segment",
    "primary_account_type",
    "relationship_status",
]


def build_preprocessor(categorical_cols: List[str], numerical_cols: List[str]) -> ColumnTransformer:
    """Build column transformer that standardizes numericals and one-hot encodes categoricals."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        verbose_feature_names_out=False,
    )


def build_pipeline_for_classifier(
    classifier: Any,
    categorical_cols: List[str],
    numerical_cols: List[str],
    n_clusters: int = 4,
    contamination: float = 0.05,
    random_state: int = 42,
) -> Pipeline:
    """
    Constructs an end-to-end Pipeline:
    1. DomainFeatureEngineer: computes ratio and friction indices.
    2. ColumnTransformer: scales numericals and encodes categoricals.
    3. UnsupervisedFeatureEngineer: fits Isolation Forest & K-Means to extract anomaly and clustering features.
    4. Classifier: final supervised estimator.
    """
    preprocessor = build_preprocessor(categorical_cols, numerical_cols)
    unsupervised = UnsupervisedFeatureEngineer(
        n_clusters=n_clusters,
        contamination=contamination,
        random_state=random_state,
    )

    return Pipeline(
        steps=[
            ("domain", DomainFeatureEngineer()),
            ("preprocessor", preprocessor),
            ("unsupervised", unsupervised),
            ("classifier", classifier),
        ]
    )


def extract_feature_names(fitted_pipeline: Pipeline) -> List[str]:
    """Retrieves full feature names output after preprocessing and unsupervised feature generation."""
    preproc = fitted_pipeline.named_steps["preprocessor"]
    base_names = list(preproc.get_feature_names_out())
    unsup = fitted_pipeline.named_steps["unsupervised"]
    return list(unsup.get_feature_names_out(base_names))


def compute_feature_importances(
    fitted_pipeline: Pipeline,
    feature_names: List[str],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42,
) -> List[Dict[str, Any]]:
    """Calculates top feature importances based on classifier type."""
    clf = fitted_pipeline.named_steps["classifier"]

    # 1. Tree-based models (Random Forest, XGBoost, LightGBM)
    if hasattr(clf, "feature_importances_"):
        raw_importances = clf.feature_importances_
        paired = [
            {
                "feature": name,
                "importance": float(abs(imp)),
                "raw_value": float(imp),
            }
            for name, imp in zip(feature_names, raw_importances)
        ]

    # 2. Linear models (Logistic Regression)
    elif hasattr(clf, "coef_"):
        coefficients = clf.coef_[0]
        paired = [
            {
                "feature": name,
                "importance": float(abs(coef)),
                "raw_value": float(coef),
                "direction": "increases_risk" if coef > 0 else "decreases_risk",
            }
            for name, coef in zip(feature_names, coefficients)
        ]

    # 3. Kernel SVC (Permutation Importance on test set sample)
    elif isinstance(clf, SVC):
        try:
            # Transform test features up to the classifier step
            X_test_prep = fitted_pipeline.named_steps["domain"].transform(X_test)
            X_test_prep = fitted_pipeline.named_steps["preprocessor"].transform(X_test_prep)
            X_test_prep = fitted_pipeline.named_steps["unsupervised"].transform(X_test_prep)

            sample_size = min(300, len(X_test))
            perm = permutation_importance(
                clf,
                X_test_prep[:sample_size],
                y_test.iloc[:sample_size],
                n_repeats=3,
                random_state=random_state,
                n_jobs=-1,
            )
            paired = [
                {
                    "feature": name,
                    "importance": float(abs(mean_imp)),
                    "raw_value": float(mean_imp),
                }
                for name, mean_imp in zip(feature_names, perm.importances_mean)
            ]
        except Exception as exc:
            warnings.warn(f"SVC permutation importance calculation failed: {exc}")
            paired = []
    else:
        paired = []

    # Sort descending by importance
    paired.sort(key=lambda x: x["importance"], reverse=True)
    return paired


def train_and_evaluate_all_models(
    df: Optional[pd.DataFrame] = None,
    random_state: int = 42,
    test_size: float = 0.2,
) -> Dict[str, Any]:
    """
    Trains and evaluates all 5 ML model roles:
    1. Logistic Regression (The Baseline)
    2. Random Forest (The Bagging Ensemble)
    3. Gradient Boosting: XGBoost
    4. Gradient Boosting: LightGBM
    5. Support Vector Classifier (The Non-Linear Classifier)
    All augmented with Unsupervised Feature Engineers (Isolation Forest & K-Means).
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    if df is None:
        df = load_processed_dataset()

    X_raw = df.drop(columns=["customer_id", "relationship_exit"], errors="ignore")
    y = df["relationship_exit"]

    # Identify columns after domain feature engineering
    df_engineered_sample = add_engineered_features(X_raw.iloc[:5])
    categorical_cols = [c for c in CATEGORICAL_FEATURES if c in df_engineered_sample.columns]
    numerical_cols = [c for c in df_engineered_sample.columns if c not in categorical_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Define model estimators with established parameters
    model_configs: Dict[str, Dict[str, Any]] = {
        "logistic_regression": {
            "display_name": "Logistic Regression",
            "role": "The Baseline",
            "estimator": LogisticRegression(
                max_iter=2000,
                random_state=random_state,
                class_weight="balanced",
            ),
        },
        "random_forest": {
            "display_name": "Random Forest",
            "role": "The Bagging Ensemble",
            "estimator": RandomForestClassifier(
                n_estimators=250,
                random_state=random_state,
                class_weight="balanced_subsample",
                n_jobs=-1,
            ),
        },
        "xgboost": {
            "display_name": "XGBoost",
            "role": "Gradient Boosting",
            "estimator": XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                random_state=random_state,
                eval_metric="logloss",
                n_jobs=-1,
            ),
        },
        "lightgbm": {
            "display_name": "LightGBM",
            "role": "Gradient Boosting",
            "estimator": LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                random_state=random_state,
                verbose=-1,
                n_jobs=-1,
            ),
        },
        "svc": {
            "display_name": "Support Vector Classifier (SVC)",
            "role": "The Non-Linear Classifier",
            "estimator": SVC(
                kernel="rbf",
                probability=True,
                random_state=random_state,
                class_weight="balanced",
            ),
        },
    }

    trained_pipelines: Dict[str, Pipeline] = {}
    evaluation_metrics: Dict[str, Dict[str, Any]] = {}
    common_feature_names: List[str] = []

    print("\n" + "=" * 70)
    print("STARTING ML MODEL PIPELINE TRAINING & EVALUATION")
    print("=" * 70)

    for key, cfg in model_configs.items():
        print(f"\n[Training] {cfg['display_name']} ({cfg['role']})...")
        pipeline = build_pipeline_for_classifier(
            classifier=cfg["estimator"],
            categorical_cols=categorical_cols,
            numerical_cols=numerical_cols,
            n_clusters=4,
            contamination=0.05,
            random_state=random_state,
        )

        pipeline.fit(X_train, y_train)
        trained_pipelines[key] = pipeline

        # Predictions on test set
        preds = pipeline.predict(X_test)
        probs = pipeline.predict_proba(X_test)[:, 1]

        # Metric calculations
        roc_auc = float(roc_auc_score(y_test, probs))
        precision = float(precision_score(y_test, preds, zero_division=0))
        recall = float(recall_score(y_test, preds, zero_division=0))
        f1 = float(f1_score(y_test, preds, zero_division=0))
        cm = confusion_matrix(y_test, preds)

        feature_names = extract_feature_names(pipeline)
        common_feature_names = feature_names

        top_features = compute_feature_importances(
            pipeline, feature_names, X_test, y_test, random_state=random_state
        )

        evaluation_metrics[key] = {
            "display_name": cfg["display_name"],
            "role": cfg["role"],
            "roc_auc": roc_auc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": {
                "matrix": cm.tolist(),
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
            },
            "top_features": top_features[:10],
            "all_features_count": len(feature_names),
        }

        print(
            f"   ✓ ROC-AUC: {roc_auc:.4f} | F1: {f1:.4f} | Recall: {recall:.4f} | Precision: {precision:.4f}"
        )
        print(f"   ✓ Confusion Matrix: TN={cm[0, 0]}, FP={cm[0, 1]}, FN={cm[1, 0]}, TP={cm[1, 1]}")

    # Determine best performing model by F1 Score (with ROC-AUC as tie-breaker)
    best_key = max(
        evaluation_metrics.keys(),
        key=lambda k: (evaluation_metrics[k]["f1"], evaluation_metrics[k]["roc_auc"]),
    )

    print("\n" + "=" * 70)
    print(f"BEST PERFORMING MODEL: {evaluation_metrics[best_key]['display_name']} (F1: {evaluation_metrics[best_key]['f1']:.4f})")
    print("=" * 70)

    # Assemble single bundle
    bundle = {
        "models": trained_pipelines,
        "metrics": evaluation_metrics,
        "best_model_name": best_key,
        "best_pipeline": trained_pipelines[best_key],
        "feature_names": common_feature_names,
        "categorical_columns": categorical_cols,
        "numerical_columns": numerical_cols,
        "unsupervised_config": {
            "isolation_forest": {"contamination": 0.05, "n_estimators": 100},
            "kmeans": {"n_clusters": 4},
        },
        "metadata": {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "random_state": random_state,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "target": "relationship_exit",
        },
    }

    # Save artifacts
    save_bundle_and_reports(bundle, evaluation_metrics)
    return bundle


def save_bundle_and_reports(bundle: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    """Exports bundle to single .pkl and reports to JSON."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Save single .pkl bundle containing all models and preprocessors
    joblib.dump(bundle, BUNDLE_PATH)
    print(f"\n[Saved] Single unified bundle .pkl: {BUNDLE_PATH}")

    # 2. Also save best pipeline as primary .pkl for direct legacy / standalone loading
    joblib.dump(bundle["best_pipeline"], PRIMARY_MODEL_PATH)
    print(f"[Saved] Best model pipeline .pkl: {PRIMARY_MODEL_PATH}")

    # 3. Save JSON evaluation metrics
    json_path = REPORTS_DIR / "model_evaluation_metrics.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[Saved] Evaluation metrics JSON: {json_path}")


def load_churn_bundle(bundle_path: Path = BUNDLE_PATH) -> Dict[str, Any]:
    """Loads the single .pkl bundle, training from scratch if missing."""
    if not bundle_path.exists():
        print("Bundle .pkl not found. Training models now...")
        return train_and_evaluate_all_models()
    return joblib.load(bundle_path)


if __name__ == "__main__":
    train_and_evaluate_all_models()
