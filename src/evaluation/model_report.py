from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "results"


def save_experiment_results(results: Dict[str, Any], name: str = "baseline_model_results.json") -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / name
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"Saved experiment results to: {output_path}")
