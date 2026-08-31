from __future__ import annotations

from typing import Dict, List

import pandas as pd


def score_risk(probabilities: pd.Series) -> pd.Series:
    risk_scores = probabilities.copy()
    return risk_scores


def assign_risk_category(probabilities: pd.Series) -> pd.Series:
    categories = []
    for prob in probabilities:
        if prob < 0.30:
            categories.append("Low")
        elif prob < 0.60:
            categories.append("Moderate")
        elif prob < 0.80:
            categories.append("High")
        else:
            categories.append("Critical")
    return pd.Series(categories, index=probabilities.index)


def build_priority_table(df: pd.DataFrame, probability_col: str = "predicted_exit_probability") -> pd.DataFrame:
    result = df.copy()
    result["risk_category"] = assign_risk_category(result[probability_col])
    result["priority_score"] = result[probability_col].astype(float) * 100.0
    result["priority_rank"] = result["priority_score"].rank(method="first", ascending=False).astype(int)
    return result.sort_values(["priority_score", probability_col], ascending=[False, False])
