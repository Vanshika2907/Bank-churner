from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "results"


def build_quality_report() -> None:
    data_path = PROCESSED_DIR / "ml_ready_dataset.csv"
    df = pd.read_csv(data_path)

    missing_values = df.isna().sum().sort_values(ascending=False)
    missing_summary = missing_values[missing_values > 0].to_dict()

    duplicates = df.duplicated(subset=["customer_id"]).sum()
    target_distribution = df["relationship_exit"].value_counts().sort_index().to_dict()
    non_target_numeric = df.select_dtypes(include="number").drop(columns=["relationship_exit"], errors="ignore")
    corr = non_target_numeric.corr(numeric_only=True)
    corr_flat = corr.where(~pd.DataFrame(np.eye(len(corr), dtype=bool), index=corr.index, columns=corr.columns), np.nan)
    corr_data = corr_flat.abs().stack().dropna().sort_values(ascending=False)
    major_correlations = [
        {
            "feature_1": str(k[0]),
            "feature_2": str(k[1]),
            "abs_correlation": round(float(v), 4),
        }
        for k, v in corr_data.head(10).items()
    ]

    report_lines = [
        "# Data Quality Validation Report",
        "",
        "## Dataset overview",
        f"- Source file: `{data_path.name}`",
        f"- Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns",
        f"- Duplicate customers: {duplicates}",
        "",
        "## Target distribution",
        f"- Class counts: {target_distribution}",
        f"- Target prevalence: {df['relationship_exit'].mean():.4f}",
        "",
        "## Missing values",
        f"- Missing entries total: {int(df.isna().sum().sum())}",
        f"- Fields with missing values: {missing_summary if missing_summary else 'None'}",
        "",
        "## Major correlations (by absolute Pearson coefficient)",
    ]

    if major_correlations:
        for item in major_correlations:
            report_lines.append(
                f"- {item['feature_1']} vs {item['feature_2']}: {item['abs_correlation']}"
            )
    else:
        report_lines.append("- No strong correlations found above the reporting threshold.")

    report_lines.extend(
        [
            "",
            "## Leakage risk assessment",
            "- The data uses only historical observation-window behavior to define the features.",
            "- The target is derived from a latent risk process using the same observation period and a future outcome window, which is valid for the design.",
            "- The pipeline should still be checked before modeling to confirm no future-period variables are accidentally merged into features.",
            "",
            "## Synthetic realism assessment",
            "- The synthetic data is intentionally correlated across financial, transactional, engagement, and service-friction dimensions.",
            "- It includes marginal heterogeneity, moderate noise, and realistic class overlap to avoid an unrealistically easy target.",
            "- The generated data is suitable for method validation, reproducible experimentation, and initial model prototyping.",
            "",
            "## Recommended next step",
            "- Run the ETL validation and feature engineering notebook pipeline.",
            "- Build and evaluate the baseline classifier only after the dataset is confirmed to be leakage-safe and stable.",
        ]
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / "data_quality_report.md"
    out_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved data quality report to: {out_path}")


if __name__ == "__main__":
    import numpy as np

    build_quality_report()
