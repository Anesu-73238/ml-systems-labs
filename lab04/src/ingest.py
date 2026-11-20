import logging
from pathlib import Path

from sklearn.datasets import fetch_openml

log = logging.getLogger("ingest")
RAW_PATH = Path("data/raw_adult.csv")
EXPECTED_SHAPE = (48842, 15)


def ingest():
    """Read the raw data exactly as delivered. No cleaning, no renaming, no dropping."""
    df = fetch_openml(name="adult", version=2, as_frame=True).frame
    log.info("Ingested rows=%d columns=%d", df.shape[0], df.shape[1])
    assert df.shape == EXPECTED_SHAPE, f"unexpected shape {df.shape}"
    RAW_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(RAW_PATH, index=False)  # untouched copy for inspection
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = ingest()
    print(df.dtypes)
    print(df.head())
    print("missing values per column:\n", df.isna().sum())