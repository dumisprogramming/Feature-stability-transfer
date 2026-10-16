from pathlib import Path
from collections import Counter, defaultdict
import zipfile
import pandas as pd
import json
import time

DATA_DIR = Path(
    "/content/drive/MyDrive/research/researchdata/"
    "NIDS_Research/Paper3_DataSet"
)

AUDIT_DIR = DATA_DIR / "Paper3_Audit" / "temporal_audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "NF_BoT_IoT_v3": "NF-BoT-IoT-v3.zip",
    "NF_CSE_CIC_IDS2018_v3": "NF-CSE-CIC-IDS2018-v3.zip",
    "NF_ToN_IoT_v3": "NF-ToN-IoT-v3.zip",
    "NF_UNSW_NB15_v3": "NF-UNSW-NB15-v3.zip",
}

USE_COLUMNS = [
    "FLOW_START_MILLISECONDS",
    "FLOW_END_MILLISECONDS",
    "Label",
    "Attack",
]

CHUNK_SIZE = 500_000

# Prespecified before examining temporal counts
BLOCK_HOURS = 6
MIN_BENIGN_PER_PERIOD = 2_000
MIN_ATTACK_PER_PERIOD = 2_000


def update_period_counter(counter, periods, labels):
    frame = pd.DataFrame({
        "period": periods,
        "label": labels,
    }).dropna()

    frame = frame[frame["label"].isin([0, 1])]

    if frame.empty:
        return

    grouped = frame.groupby(["period", "label"]).size()

    for (period, label), count in grouped.items():
        counter[str(period)][int(label)] += int(count)


def period_table(counter, period_type):
    rows = []

    for period in sorted(counter):
        benign = int(counter[period].get(0, 0))
        attack = int(counter[period].get(1, 0))

        eligible = (
            benign >= MIN_BENIGN_PER_PERIOD
            and attack >= MIN_ATTACK_PER_PERIOD
        )

        rows.append({
            "period_type": period_type,
            "period": period,
            "benign": benign,
            "attack": attack,
            "total": benign + attack,
            "eligible": eligible,
        })

    return pd.DataFrame(rows)


all_summaries = []

print("=" * 78)
print("PAPER 3 — STREAMING TEMPORAL AND LABEL AUDIT")
print("=" * 78)

for dataset_name, zip_filename in DATASETS.items():

    checkpoint = AUDIT_DIR / f"{dataset_name}_audit.json"

    # Resume support
    if checkpoint.exists():
        with open(checkpoint, "r", encoding="utf-8") as file:
            existing = json.load(file)

        all_summaries.append(existing)
        print(f"\nSKIPPED—checkpoint exists: {dataset_name}")
        continue

    zip_path = DATA_DIR / zip_filename

    print(f"\n{'-' * 78}")
    print(f"DATASET: {dataset_name}")
    print(f"ZIP: {zip_path}")
    print(f"{'-' * 78}")

    if not zip_path.exists():
        print("ERROR: ZIP file not found.")
        continue

    row_count = 0
    invalid_start_timestamp = 0
    invalid_end_timestamp = 0
    invalid_label_count = 0

    label_counts = Counter()
    attack_counts = Counter()
    daily_counts = defaultdict(Counter)
    block_counts = defaultdict(Counter)

    minimum_time = None
    maximum_time = None

    start_time = time.time()

    with zipfile.ZipFile(zip_path, "r") as archive:

        candidate_members = [
            member for member in archive.namelist()
            if member.lower().endswith(".csv")
            and "netflow_v3_features" not in member.lower()
            and not member.startswith("__MACOSX")
        ]

        if not candidate_members:
            raise RuntimeError(
                f"No principal dataset CSV found in {zip_filename}"
            )

        # Select the largest CSV as the principal dataset
        csv_member = max(
            candidate_members,
            key=lambda member: archive.getinfo(member).file_size,
        )

        print(f"CSV member: {csv_member}")

        with archive.open(csv_member) as csv_file:

            reader = pd.read_csv(
                csv_file,
                usecols=USE_COLUMNS,
                chunksize=CHUNK_SIZE,
                low_memory=False,
            )

            for chunk_number, chunk in enumerate(reader, start=1):

                rows_in_chunk = len(chunk)
                row_count += rows_in_chunk

                start_numeric = pd.to_numeric(
                    chunk["FLOW_START_MILLISECONDS"],
                    errors="coerce",
                )

                end_numeric = pd.to_numeric(
                    chunk["FLOW_END_MILLISECONDS"],
                    errors="coerce",
                )

                labels = pd.to_numeric(
                    chunk["Label"],
                    errors="coerce",
                )

                start_datetime = pd.to_datetime(
                    start_numeric,
                    unit="ms",
                    utc=True,
                    errors="coerce",
                )

                end_datetime = pd.to_datetime(
                    end_numeric,
                    unit="ms",
                    utc=True,
                    errors="coerce",
                )

                invalid_start_timestamp += int(
                    start_datetime.isna().sum()
                )

                invalid_end_timestamp += int(
                    end_datetime.isna().sum()
                )

                invalid_label_count += int(
                    (~labels.isin([0, 1])).sum()
                )

                valid_labels = labels[labels.isin([0, 1])].astype(int)

                label_counts.update(
                    valid_labels.value_counts().to_dict()
                )

                attack_values = (
                    chunk["Attack"]
                    .fillna("__MISSING__")
                    .astype(str)
                    .str.strip()
                )

                attack_counts.update(
                    attack_values.value_counts().to_dict()
                )

                valid_times = start_datetime.dropna()

                if not valid_times.empty:
                    chunk_minimum = valid_times.min()
                    chunk_maximum = valid_times.max()

                    if (
                        minimum_time is None
                        or chunk_minimum < minimum_time
                    ):
                        minimum_time = chunk_minimum

                    if (
                        maximum_time is None
                        or chunk_maximum > maximum_time
                    ):
                        maximum_time = chunk_maximum

                days = start_datetime.dt.strftime("%Y-%m-%d")

                blocks = (
                    start_datetime
                    .dt.floor(f"{BLOCK_HOURS}h")
                    .dt.strftime("%Y-%m-%d %H:%M:%S")
                )

                update_period_counter(
                    daily_counts,
                    days,
                    labels,
                )

                update_period_counter(
                    block_counts,
                    blocks,
                    labels,
                )

                if chunk_number % 5 == 0:
                    elapsed_minutes = (
                        time.time() - start_time
                    ) / 60

                    print(
                        f"  Chunks: {chunk_number:,} | "
                        f"Rows: {row_count:,} | "
                        f"Elapsed: {elapsed_minutes:.1f} min"
                    )

    daily_table = period_table(
        daily_counts,
        "calendar_day",
    )

    block_table = period_table(
        block_counts,
        f"{BLOCK_HOURS}_hour_block",
    )

    daily_file = AUDIT_DIR / f"{dataset_name}_daily_counts.csv"
    block_file = AUDIT_DIR / f"{dataset_name}_6hour_counts.csv"

    daily_table.to_csv(daily_file, index=False)
    block_table.to_csv(block_file, index=False)

    elapsed_minutes = (time.time() - start_time) / 60

    summary = {
        "dataset": dataset_name,
        "zip_file": zip_filename,
        "csv_member": csv_member,
        "total_rows": int(row_count),
        "benign_label_0": int(label_counts.get(0, 0)),
        "attack_label_1": int(label_counts.get(1, 0)),
        "invalid_labels": int(invalid_label_count),
        "invalid_start_timestamps": int(
            invalid_start_timestamp
        ),
        "invalid_end_timestamps": int(
            invalid_end_timestamp
        ),
        "minimum_start_time": (
            minimum_time.isoformat()
            if minimum_time is not None else None
        ),
        "maximum_start_time": (
            maximum_time.isoformat()
            if maximum_time is not None else None
        ),
        "calendar_days": int(len(daily_table)),
        "eligible_calendar_days": int(
            daily_table["eligible"].sum()
            if not daily_table.empty else 0
        ),
        "six_hour_blocks": int(len(block_table)),
        "eligible_six_hour_blocks": int(
            block_table["eligible"].sum()
            if not block_table.empty else 0
        ),
        "attack_category_counts": {
            str(key): int(value)
            for key, value in attack_counts.items()
        },
        "elapsed_minutes": round(elapsed_minutes, 2),
    }

    with open(checkpoint, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    all_summaries.append(summary)

    print(f"\nCOMPLETED: {dataset_name}")
    print(f"Rows: {row_count:,}")
    print(f"Label 0—benign: {label_counts.get(0, 0):,}")
    print(f"Label 1—attack: {label_counts.get(1, 0):,}")
    print(f"Calendar days: {len(daily_table)}")
    print(
        "Eligible calendar days: "
        f"{daily_table['eligible'].sum()}"
    )
    print(f"Six-hour blocks: {len(block_table)}")
    print(
        "Eligible six-hour blocks: "
        f"{block_table['eligible'].sum()}"
    )
    print(f"Elapsed: {elapsed_minutes:.1f} minutes")


summary_table = pd.DataFrame([
    {
        "dataset": item["dataset"],
        "total_rows": item["total_rows"],
        "benign": item["benign_label_0"],
        "attack": item["attack_label_1"],
        "invalid_labels": item["invalid_labels"],
        "invalid_start_times": item[
            "invalid_start_timestamps"
        ],
        "calendar_days": item["calendar_days"],
        "eligible_days": item["eligible_calendar_days"],
        "six_hour_blocks": item["six_hour_blocks"],
        "eligible_6h_blocks": item[
            "eligible_six_hour_blocks"
        ],
    }
    for item in all_summaries
])

summary_path = AUDIT_DIR / "combined_temporal_audit.csv"
summary_table.to_csv(summary_path, index=False)

print("\n" + "=" * 78)
print("COMBINED TEMPORAL AUDIT")
print("=" * 78)
print(summary_table.to_string(index=False))
print(f"\nSaved: {summary_path}")