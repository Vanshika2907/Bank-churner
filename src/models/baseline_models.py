from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def train_baseline_models(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=random_state,
    )

    models: List[Tuple[str, object]] = [
        ("dummy", DummyClassifier(strategy="most_frequent")),
        ("logistic_regression", LogisticRegression(max_iter=2000, random_state=random_state)),
        ("decision_tree", DecisionTreeClassifier(random_state=random_state, max_depth=6)),
        ("random_forest", RandomForestClassifier(n_estimators=250, random_state=random_state, class_weight="balanced_subsample")),
        ("gradient_boosting", GradientBoostingClassifier(random_state=random_state)),
    ]

    results: Dict[str, Dict[str, float]] = {}
    for name, model in models:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "f1": float(f1_score(y_test, preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probs)),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        }

    return results
