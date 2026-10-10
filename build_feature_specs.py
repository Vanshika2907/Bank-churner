"""
Run once from the project root:  python build_feature_specs.py

Reads the data your model was trained on and writes config/feature_specs.json,
which app.py uses for each input's min, max and default.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

# DATA_PATH comes from predictor.py (data/processed/ml_ready_dataset.csv), so the
# input ranges always come from the exact file the model is trained on.
from src.model.predictor import DATA_PATH, FEATURE_GROUPS

OUT_PATH = Path("config/feature_specs.json")


def nice_step(span: float, integer: bool) -> float:
    """Roughly span/100, rounded down to a power of ten."""
    raw = span / 100
    step = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    return float(max(1, int(step))) if integer else float(step)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    fields = [f for group in FEATURE_GROUPS.values() for f in group]

    missing = [f for f in fields if f not in df.columns]
    if missing:
        print("These fields are not columns in the data file:")
        for f in missing:
            print("  -", f)
        print("Check DATA_PATH, or whether the file uses different column names.")
        sys.exit(1)

    specs: dict[str, dict] = {}
    for f in fields:
        s = df[f].dropna()

        if not pd.api.types.is_numeric_dtype(s):
            specs[f] = {"type": "categorical", "options": sorted(s.unique().tolist())}
            continue

        lo, hi = float(s.min()), float(s.max())
        if hi == lo:
            print(f"WARNING: {f} is constant ({lo}); the model cannot learn from it")
            hi = lo + 1.0
        is_int = bool((s % 1 == 0).all())
        default = float(s.median())
        step = nice_step(hi - lo, is_int)

        if is_int:
            specs[f] = {"type": "int", "min": int(lo), "max": int(hi),
                        "default": int(round(default)), "step": int(step)}
        else:
            specs[f] = {"type": "float", "min": lo, "max": hi,
                        "default": default, "step": step}

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        json.dump(specs, fh, indent=2)

    # Print a summary so you can sanity-check every field
    print(f"Wrote {OUT_PATH}\n")
    for f, s in specs.items():
        if s["type"] == "categorical":
            print(f"{f:45s} categorical  {s['options']}")
        else:
            print(f"{f:45s} {s['type']:5s}  min={s['min']:.4g}  max={s['max']:.4g}  default={s['default']:.4g}")


if __name__ == "__main__":
    main()