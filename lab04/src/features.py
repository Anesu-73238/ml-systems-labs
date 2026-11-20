import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preprocess import CAT_COLS, NUM_COLS, TARGET, load_config

log = logging.getLogger("features")
POS = ">50K"


def add_features(df):
    """Row-wise derived features: only use values in the same row, so no leakage."""
    assert (df["age"] > 16).all(), "age - 16 must be positive"  # age is clipped to >= 17
    df = df.copy()
    df["capital_net"] = df["capital-gain"] - df["capital-loss"]
    df["hours_per_age"] = df["hours-per-week"] / (df["age"] - 16)
    return df


def evaluate(train, test, extra):
    """Fit scaler, one-hot encoder and logistic regression on TRAIN only; score on TEST."""
    num = NUM_COLS + extra
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])
    model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])
    model.fit(train[num + CAT_COLS], train[TARGET])
    pred = model.predict(test[num + CAT_COLS])
    return {
        "accuracy": round(float(accuracy_score(test[TARGET], pred)), 4),
        "f1": round(float(f1_score(test[TARGET], pred, pos_label=POS)), 4),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    proc = Path(cfg["paths"]["processed_dir"])
    train = add_features(pd.read_csv(proc / "train.csv"))
    test = add_features(pd.read_csv(proc / "test.csv"))

    configs = {
        "baseline (no derived)": [],
        "+ capital_net": ["capital_net"],
        "+ hours_per_age": ["hours_per_age"],
        "+ both": ["capital_net", "hours_per_age"],
    }
    results = {name: evaluate(train, test, extra) for name, extra in configs.items()}

    print(f"{'features':25s} {'accuracy':>9s} {'F1 (>50K)':>10s}")
    for name, r in results.items():
        print(f"{name:25s} {r['accuracy']:9.4f} {r['f1']:10.4f}")
    Path("results/feature_eval.json").write_text(json.dumps(results, indent=2))
    print("saved results/feature_eval.json")