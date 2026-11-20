import logging
from pathlib import Path

import joblib
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

log = logging.getLogger("preprocess")

NUM_COLS = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CAT_COLS = ["workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country"]
TARGET = "income"


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


class Cleaner(BaseEstimator, TransformerMixin):
    """Median/mode imputation + percentile clipping. Everything is learned in fit()
    (call it on TRAIN only) and then reused unchanged by transform()."""

    def __init__(self, num_cols, cat_cols, quantiles=(0.01, 0.99)):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.quantiles = quantiles

    def fit(self, X, y=None):
        self.medians_ = X[self.num_cols].median()
        self.modes_ = {c: X[c].mode(dropna=True).iloc[0] for c in self.cat_cols}
        imputed = X[self.num_cols].fillna(self.medians_)
        lo, hi = self.quantiles
        self.lower_ = imputed.quantile(lo)
        self.upper_ = imputed.quantile(hi)
        return self

    def transform(self, X):
        X = X.copy()
        X[self.num_cols] = (X[self.num_cols].fillna(self.medians_)
                            .clip(self.lower_, self.upper_, axis=1))
        for c in self.cat_cols:
            X[c] = X[c].fillna(self.modes_[c])
        return X


def split_and_clean(df, cfg):
    # 1. split BEFORE any cleaning, stratified on the label
    train, test = train_test_split(
        df, test_size=cfg["test_size"], stratify=df[TARGET], random_state=cfg["seed"])
    stats = {"train_rows_before_dedup": len(train), "test_rows": len(test)}

    # 2. exact duplicates are dropped from train only; test stays untouched
    train = train.drop_duplicates()
    stats["train_duplicates_dropped"] = stats["train_rows_before_dedup"] - len(train)

    # 3. fit on train only
    cleaner = Cleaner(NUM_COLS, CAT_COLS, tuple(cfg["clip_quantiles"]))
    cleaner.fit(train.drop(columns=TARGET))

    # 4. apply the train-fitted cleaner to both parts
    out = {}
    for name, part in [("train", train), ("test", test)]:
        X = cleaner.transform(part.drop(columns=TARGET))
        X[TARGET] = part[TARGET].to_numpy()
        out[name] = X.reset_index(drop=True)

    assert out["train"].isna().sum().sum() == 0 and out["test"].isna().sum().sum() == 0
    assert len(out["test"]) == stats["test_rows"]  # no test rows dropped
    stats["train_rows"] = len(out["train"])

    Path(cfg["paths"]["preprocessor"]).parent.mkdir(exist_ok=True)
    joblib.dump(cleaner, cfg["paths"]["preprocessor"])
    proc = Path(cfg["paths"]["processed_dir"])
    proc.mkdir(parents=True, exist_ok=True)
    out["train"].to_csv(proc / "train.csv", index=False)
    out["test"].to_csv(proc / "test.csv", index=False)
    return out, cleaner, stats


if __name__ == "__main__":
    from src.ingest import ingest
    from src.validate import validate

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    df, _ = validate(ingest())
    out, cleaner, stats = split_and_clean(df, cfg)
    print(stats)
    print("\nlearned on train only:")
    print("medians:\n", cleaner.medians_)
    print("modes:", cleaner.modes_)
    print("clip lower:\n", cleaner.lower_)
    print("clip upper:\n", cleaner.upper_)
    for name in ("train", "test"):
        print(f"\n{name}: {out[name].shape}, positive rate "
              f"{(out[name][TARGET] == '>50K').mean():.3f}")