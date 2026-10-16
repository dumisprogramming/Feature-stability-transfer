from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(
    "/content/drive/MyDrive/research/researchdata/"
    "NIDS_Research/Paper3_DataSet"
)

EXPECTED_A2 = (
    "2217078acfa10700f4fae29c6f32f171"
    "ce7a4e2ecb22d22ce5eabb6087e5aa5e"
)

A2_PATH = (
    ROOT / "Paper3_Protocol" /
    "amendment_A2_target_evaluation_plan.json"
)

def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()

print("=" * 78)
print("PAPER 3 — TARGET-STAGE READ-ONLY PREFLIGHT")
print("=" * 78)

assert A2_PATH.exists(), f"Missing A2: {A2_PATH}"

observed_a2 = sha256_file(A2_PATH)
print("A2 path:", A2_PATH)
print("Expected A2 SHA-256:", EXPECTED_A2)
print("Observed A2 SHA-256:", observed_a2)
print("A2 verification:", "PASS" if observed_a2 == EXPECTED_A2 else "FAIL")

assert observed_a2 == EXPECTED_A2, "A2 hash mismatch"

with open(A2_PATH, "r", encoding="utf-8") as f:
    a2 = json.load(f)

print("\nA2 top-level keys:")
print(sorted(a2.keys()))

print("\nRelevant directories:")
for name in [
    "Paper3_Protocol",
    "Paper3_Materialized",
    "Paper3_Sanitized",
    "Paper3_Source_Models",
    "Paper3_Source_Stability",
]:
    path = ROOT / name
    print(f"{name}: {'FOUND' if path.exists() else 'MISSING'}")

sanitized_root = ROOT / "Paper3_Sanitized"
parquet_files = sorted(sanitized_root.rglob("*.parquet"))
csv_files = sorted(sanitized_root.rglob("*.csv"))
json_files = sorted(sanitized_root.rglob("*.json"))

print("\nSanitized inventory:")
print("Parquet files:", len(parquet_files))
print("CSV files:", len(csv_files))
print("JSON files:", len(json_files))

print("\nSanitized Parquet paths:")
for path in parquet_files:
    try:
        frame = pd.read_parquet(path)
        print("-" * 78)
        print("PATH:", path.relative_to(ROOT))
        print("ROWS:", len(frame))
        print("COLUMNS:", list(frame.columns))
    except Exception as exc:
        print("READ FAILURE:", path.relative_to(ROOT), repr(exc))

print("\nSanitized CSV paths:")
for path in csv_files:
    print(" ", path.relative_to(ROOT))

print("\nSanitized JSON paths:")
for path in json_files:
    print(" ", path.relative_to(ROOT))

print("\nSource metrics schema:")
source_metrics = (
    ROOT / "Paper3_Source_Models" /
    "source_test_metrics_all_models.csv"
)
if source_metrics.exists():
    df = pd.read_csv(source_metrics, nrows=5)
    print("PATH:", source_metrics)
    print("COLUMNS:", list(df.columns))
    print(df.to_string(index=False))
else:
    print("MISSING:", source_metrics)

print("\nWarning-score schema:")
warning_scores = (
    ROOT / "Paper3_Source_Stability" /
    "source_only_warning_scores.csv"
)
if warning_scores.exists():
    df = pd.read_csv(warning_scores, nrows=5)
    print("PATH:", warning_scores)
    print("COLUMNS:", list(df.columns))
    print(df.to_string(index=False))
else:
    print("MISSING:", warning_scores)

print("\nTARGET-STAGE READ-ONLY PREFLIGHT: PASS")
print("No target panels, predictions, or performance results were created.")
print("=" * 78)