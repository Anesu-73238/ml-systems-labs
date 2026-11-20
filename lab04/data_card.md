# Data Card: UCI Adult (Census Income), Lab 4 pipeline

## Source
- UCI Machine Learning Repository, "Adult" dataset (Becker & Kohavi, 1996), extracted from the 1994 US Census database.
- Loaded with `fetch_openml(name='adult', version=2, as_frame=True)`.
- Task: predict whether annual income exceeds $50K (`>50K` vs `<=50K`).

## License
- CC BY 4.0, as listed by the UCI repository. Attribution to the original authors is required.

## Size
- Raw: 48,842 rows, 15 columns (14 features + `income` target).
- Split: 70/30 stratified before any cleaning. Test set: 14,653 rows, positive rate 0.239.
- Train set has exact duplicates removed after the split. The test set is untouched.
- Processed-data SHA256: see `results/processed_hash.json` (written by `python -m src.run_all`).

## Processing summary
- Missing values are marked `?` (workclass, occupation, native-country). They are kept as a category in validation, then imputed in the pipeline: mode for categorical, median for numeric, fitted on train only.
- Numeric outliers are clipped to the train 1st/99th percentiles.
- Derived features: `capital_net`, `hours_per_age`.
- Fitted preprocessor: `results/preprocessor.joblib`. Settings: `config.yaml`.

## Known biases
- Snapshot of 1994 US data. Income thresholds, occupations and norms are outdated, and $50K is not inflation-adjusted.
- Demographic imbalance: most rows are male, White, and US-born. Positive-label rates differ strongly by sex and race, so models can learn and amplify these gaps.
- `fnlwgt` is a census sampling weight, not a personal attribute, and it is not representative of individuals.
- Label noise: the duplicate-conflict audit (`results/label_audit.json`) finds identical feature rows with different income labels, which caps achievable accuracy.
- Missing values are not random: `?` is concentrated in workclass and occupation.

## Privacy concerns
- No names or direct identifiers, but age, sex, race, native-country, occupation and marital status together are quasi-identifiers and carry some re-identification risk.
- Contains sensitive attributes (race, sex, native-country). Do not use them to make real decisions about individuals.
- Models trained on this data should not be used for hiring, lending or other consequential decisions.

## Reproducibility
- One command: `python -m src.run_all`
- Fixed seed in `config.yaml`; every step reads its settings from that file.
