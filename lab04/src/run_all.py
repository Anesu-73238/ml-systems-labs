
import hashlib
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("run_all")

STEPS = ["src.ingest", "src.validate", "src.preprocess", "src.features", "src.label_audit"]
PROCESSED_DIR = Path("data/processed")
RESULTS = Path("results")
CONFIG = Path("config.yaml")


def run_step(module: str) -> None:
    log.info("Running %s", module)
    proc = subprocess.run([sys.executable, "-m", module])
    if proc.returncode != 0:
        raise SystemExit(f"Step {module} failed with exit code {proc.returncode}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_processed() -> dict:
    files = sorted(p for p in PROCESSED_DIR.rglob("*") if p.is_file())
    if not files:
        raise SystemExit(f"No files found in {PROCESSED_DIR}; nothing to hash.")
    per_file = {p.relative_to(PROCESSED_DIR).as_posix(): sha256_file(p) for p in files}
    # Combined hash: depends on file names and contents, in sorted order
    combined = hashlib.sha256(
        "".join(f"{name}:{digest}\n" for name, digest in per_file.items()).encode()
    ).hexdigest()
    return {"algorithm": "sha256", "combined": combined, "files": per_file}


def main() -> None:
    for module in STEPS:
        run_step(module)

    RESULTS.mkdir(exist_ok=True)
    digest = hash_processed()
    (RESULTS / "processed_hash.json").write_text(json.dumps(digest, indent=2))
    shutil.copy(CONFIG, RESULTS / "config_used.yaml")
    log.info("Processed-data SHA256: %s", digest["combined"])
    log.info("Saved results/processed_hash.json and results/config_used.yaml")


if __name__ == "__main__":
    main()
