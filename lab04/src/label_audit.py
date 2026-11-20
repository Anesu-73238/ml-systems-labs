import json
import logging
from pathlib import Path

import pandas as pd

from src.preprocess import CAT_COLS, NUM_COLS, TARGET, load_config

log = logging.getLogger("label_audit")
FEATURES = NUM_COLS + CAT_COLS  # the 14 input features


def conflict_audit(df, features):
    """Rows with identical feature values but different income labels."""
    d = df[features + [TARGET]].copy()
    for c in features:
        if c in CAT_COLS:
            d[c] = d[c].astype(object).fillna("?")
    d[TARGET] = d[TARGET].astype(str)
    g = d.groupby(features, sort=False)[TARGET].agg(size="size", n_labels="nunique")
    dup = g[g["size"] > 1]
    conf = dup[dup["n_labels"] > 1]
    return {
        "features_used": len(features),
        "duplicate_groups": int(len(dup)),
        "rows_in_duplicate_groups": int(dup["size"].sum()),
        "conflicting_groups": int(len(conf)),
        "rows_in_conflicting_groups": int(conf["size"].sum()),
    }, conf.reset_index()


def rule_checks(df):
    """Automated 'impossible / suspicious value' rules based on the data dictionary."""
    edu = df["education"].astype(object)
    mapping = df.groupby(edu)["education-num"].agg(lambda s: s.mode().iloc[0])
    rel, sex = df["relationship"].astype(object), df["sex"].astype(object)
    rules = {
        "education_vs_education-num_mismatch": df["education-num"] != edu.map(mapping),
        "husband_but_female": (rel == "Husband") & (sex == "Female"),
        "wife_but_male": (rel == "Wife") & (sex == "Male"),
        "degree_level_education_but_age<=19": (df["education-num"] >= 13) & (df["age"] <= 19),
        "capital_gain_and_loss_both_positive": (df["capital-gain"] > 0) & (df["capital-loss"] > 0),
    }
    return {k: int(v.sum()) for k, v in rules.items()}


if __name__ == "__main__":
    from src.ingest import ingest
    from src.validate import validate

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    df, _ = validate(ingest())  # raw data with label renamed to 'income'; nothing dropped

    summary = {}
    summary["conflicts_all_14_features"], conf_all = conflict_audit(df, FEATURES)
    summary["conflicts_without_fnlwgt"], conf_nofw = conflict_audit(
        df, [c for c in FEATURES if c != "fnlwgt"])
    summary["rule_checks_full_data"] = rule_checks(df)

    # 50 random rows for manual re-check against the data dictionary
    sample = df.sample(50, random_state=cfg["seed"]).copy()
    sample["reviewer_notes"] = ""
    summary["sample_rule_flags"] = rule_checks(sample)
    Path("results").mkdir(exist_ok=True)
    sample.to_csv("results/label_audit_sample.csv")
    conf_all.to_csv("results/label_conflicts_all14.csv", index=False)
    conf_nofw.to_csv("results/label_conflicts_no_fnlwgt.csv", index=False)
    Path("results/label_audit.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print("\n50 random rows (index = row number in the raw data):")
    print(sample.drop(columns="reviewer_notes").to_string())