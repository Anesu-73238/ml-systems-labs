import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype

log = logging.getLogger("validate")
REPORT_PATH = Path("results/validation_report.json")

SCHEMA = {
    "age": "int", "workclass": "category", "fnlwgt": "int", "education": "category",
    "education-num": "int", "marital-status": "category", "occupation": "category",
    "relationship": "category", "race": "category", "sex": "category",
    "capital-gain": "int", "capital-loss": "int", "hours-per-week": "int",
    "native-country": "category", "income": "category",
}
RANGES = {
    "age": (17, 90), "hours-per-week": (1, 99),
    "capital-gain": (0, 99999), "capital-loss": (0, 4356),
}
LABELS = {"<=50K", ">50K"}


def standardize(df):
    """Validation view of the raw data: label column renamed to 'income' and '?' turned
    into NaN, so both missing-value encodings count the same. The raw frame is not changed."""
    out = df.rename(columns={"class": "income"}).copy()
    for col, kind in SCHEMA.items():
        if kind == "category" and col in out:
            s = out[col].astype(object)
            out[col] = s.where(s != "?", np.nan).astype("category")
    return out


def validate(raw):
    n_before = len(raw)
    question_marks = {c: int((raw[c].astype(object) == "?").sum())
                      for c in raw.columns if not is_integer_dtype(raw[c])}
    df = standardize(raw)
    report = {"rows": n_before, "columns": df.shape[1], "violations": []}

    def violate(kind, detail):
        report["violations"].append({"check": kind, **detail})
        log.warning("VIOLATION %s %s", kind, detail)

    # 1. schema: names and types
    missing_cols = [c for c in SCHEMA if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in SCHEMA]
    if missing_cols or extra_cols:
        violate("columns", {"missing": missing_cols, "extra": extra_cols})
    report["types"] = {}
    for col, expected in SCHEMA.items():
        if col not in df:
            continue
        ok = is_integer_dtype(df[col]) if expected == "int" else isinstance(df[col].dtype, pd.CategoricalDtype)
        report["types"][col] = {"expected": expected, "actual": str(df[col].dtype), "ok": bool(ok)}
        if not ok:
            violate("type", {"column": col, "expected": expected, "actual": str(df[col].dtype)})

    # 2. ranges
    report["ranges"] = {}
    for col, (lo, hi) in RANGES.items():
        below, above = int((df[col] < lo).sum()), int((df[col] > hi).sum())
        report["ranges"][col] = {"allowed": [lo, hi], "min": int(df[col].min()),
                                 "max": int(df[col].max()), "below": below, "above": above}
        if below or above:
            violate("range", {"column": col, "below": below, "above": above})

    # 3. missing values ('?' and NaN both mean missing; allowed, reported, never dropped)
    report["missing"] = {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()}
    report["raw_question_mark_strings"] = {c: n for c, n in question_marks.items() if n}

    # 4. label values
    bad_labels = sorted(set(df["income"].dropna().astype(str)) - LABELS)
    report["label_counts"] = {str(k): int(v) for k, v in df["income"].value_counts(dropna=False).items()}
    if bad_labels or df["income"].isna().any():
        violate("label", {"unexpected_values": bad_labels, "missing": int(df["income"].isna().sum())})

    # 5. exact duplicate rows (reported only, not dropped here)
    report["exact_duplicate_rows"] = int(df.duplicated().sum())

    # nothing was dropped
    assert len(df) == n_before
    report["n_violations"] = len(report["violations"])
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    return df, report


if __name__ == "__main__":
    from src.ingest import ingest

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _, rep = validate(ingest())
    print(json.dumps(rep, indent=2))
    print(f"\n{rep['n_violations']} violation(s). Report saved to {REPORT_PATH}")