from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
import zipfile
import pandas as pd
import json

DATA_DIR = Path(
    "/content/drive/MyDrive/research/researchdata/NIDS_Research/Paper3_DataSet"
)

EXPECTED_FILES = [
    "NF-BoT-IoT-v3.zip",
    "NF-CSE-CIC-IDS2018-v3.zip",
    "NF-ToN-IoT-v3.zip",
    "NF-UNSW-NB15-v3.zip",
]

audit_results = {}

print("=" * 70)
print("PAPER 3 — DATASET INVENTORY AND HEADER AUDIT")
print("=" * 70)

for filename in EXPECTED_FILES:
    zip_path = DATA_DIR / filename

    if not zip_path.exists():
        print(f"\nMISSING: {filename}")
        audit_results[filename] = {"status": "missing"}
        continue

    print(f"\nFOUND: {filename}")
    print(f"ZIP size: {zip_path.stat().st_size / (1024**3):.3f} GB")

    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            bad_file = archive.testzip()

            if bad_file is not None:
                print(f"CORRUPTED MEMBER: {bad_file}")
                audit_results[filename] = {
                    "status": "corrupted",
                    "bad_member": bad_file,
                }
                continue

            csv_members = [
                item for item in archive.namelist()
                if item.lower().endswith(".csv")
                and not item.startswith("__MACOSX")
            ]

            print(f"CSV members: {len(csv_members)}")

            member_results = []

            for member in csv_members:
                info = archive.getinfo(member)

                print(f"\n  Member: {member}")
                print(
                    f"  Uncompressed size: "
                    f"{info.file_size / (1024**3):.3f} GB"
                )

                with archive.open(member) as csv_file:
                    sample = pd.read_csv(
                        csv_file,
                        nrows=5,
                        low_memory=False,
                    )

                columns = sample.columns.tolist()

                timestamp_columns = [
                    column for column in columns
                    if "TIME" in column.upper()
                    or "DATE" in column.upper()
                ]

                label_columns = [
                    column for column in columns
                    if column.upper() in {
                        "LABEL",
                        "ATTACK",
                        "BINARY_LABEL",
                        "ATTACK_LABEL",
                        "CLASS",
                    }
                ]

                print(f"  Number of columns: {len(columns)}")
                print(f"  Timestamp candidates: {timestamp_columns}")
                print(f"  Label candidates: {label_columns}")
                print(f"  Columns: {columns}")

                member_results.append({
                    "member": member,
                    "uncompressed_bytes": info.file_size,
                    "number_of_columns": len(columns),
                    "columns": columns,
                    "timestamp_candidates": timestamp_columns,
                    "label_candidates": label_columns,
                })

            audit_results[filename] = {
                "status": "pass",
                "zip_bytes": zip_path.stat().st_size,
                "members": member_results,
            }

    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}")
        audit_results[filename] = {
            "status": "error",
            "error": str(error),
        }

AUDIT_DIR = DATA_DIR / "Paper3_Audit"
AUDIT_DIR.mkdir(exist_ok=True)

output_file = AUDIT_DIR / "dataset_inventory_audit.json"

with open(output_file, "w", encoding="utf-8") as file:
    json.dump(audit_results, file, indent=2)

print("\n" + "=" * 70)
print(f"AUDIT SAVED: {output_file}")
print("=" * 70)