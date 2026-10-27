from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "speech"
    / "torgo_metadata.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "speech"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "torgo_split.csv"
)

print("=" * 70)
print("SPEECH STEP 4 - SPEAKER-AWARE SPLIT")
print("=" * 70)

df = pd.read_csv(INPUT_PATH)

print("\nTotal samples:")
print(len(df))

print("\nTotal speakers:")
print(df["speaker_id"].nunique())

# ============================================================
# FIXED SPEAKER SPLIT
# ============================================================

# IMPORTANT:
# One speaker must appear in ONE split only.

TRAIN_SPEAKERS = [
    # Dysarthria
    "F03",
    "F04",
    "M01",
    "M02",
    "M03",

    # Healthy
    "FC02",
    "FC03",
    "MC01",
    "MC03",
]

VALIDATION_SPEAKERS = [
    # Dysarthria
    "F01",
    "M04",

    # Healthy
    "MC02",
]

TEST_SPEAKERS = [
    # Dysarthria
    "M05",

    # Healthy
    "FC01",
    "MC04",
]

# ============================================================
# VALIDATE SPEAKER LIST
# ============================================================

all_assigned = (
    TRAIN_SPEAKERS
    + VALIDATION_SPEAKERS
    + TEST_SPEAKERS
)

dataset_speakers = set(
    df["speaker_id"].unique()
)

assigned_speakers = set(all_assigned)

if len(all_assigned) != len(assigned_speakers):
    raise RuntimeError(
        "A speaker appears in more than one split."
    )

missing = dataset_speakers - assigned_speakers
unknown = assigned_speakers - dataset_speakers

if missing:
    raise RuntimeError(
        f"Speakers not assigned: {sorted(missing)}"
    )

if unknown:
    raise RuntimeError(
        f"Unknown speakers: {sorted(unknown)}"
    )

# ============================================================
# ASSIGN SPLIT
# ============================================================

def get_split(speaker_id):

    if speaker_id in TRAIN_SPEAKERS:
        return "train"

    if speaker_id in VALIDATION_SPEAKERS:
        return "validation"

    if speaker_id in TEST_SPEAKERS:
        return "test"

    raise ValueError(
        f"Speaker not assigned: {speaker_id}"
    )


df["split"] = (
    df["speaker_id"]
    .apply(get_split)
)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SPLIT SUMMARY")
print("=" * 70)

for split_name in [
    "train",
    "validation",
    "test"
]:

    part = df[
        df["split"] == split_name
    ]

    print("\n" + "-" * 70)
    print(split_name.upper())
    print("-" * 70)

    print("\nSamples:")
    print(len(part))

    print("\nSpeakers:")
    print(part["speaker_id"].nunique())

    print("\nSpeaker IDs:")
    print(
        sorted(
            part["speaker_id"]
            .unique()
            .tolist()
        )
    )

    print("\nSamples by class:")
    print(
        part["speech_status"]
        .value_counts()
    )

    print("\nSpeakers by class:")

    speaker_labels = (
        part[
            ["speaker_id", "speech_status"]
        ]
        .drop_duplicates()
    )

    print(
        speaker_labels[
            "speech_status"
        ].value_counts()
    )

    print("\nGender:")
    print(
        part[
            ["speaker_id", "gender"]
        ]
        .drop_duplicates()
        ["gender"]
        .value_counts()
    )

# ============================================================
# LEAKAGE CHECK
# ============================================================

print("\n" + "=" * 70)
print("LEAKAGE CHECK")
print("=" * 70)

speaker_split_count = (
    df
    .groupby("speaker_id")["split"]
    .nunique()
)

leaked = speaker_split_count[
    speaker_split_count > 1
]

print("\nSpeakers appearing in multiple splits:")
print(len(leaked))

if len(leaked) != 0:
    raise RuntimeError(
        "SPEAKER LEAKAGE DETECTED"
    )

print("PASS - no speaker leakage.")

# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 70)
print("SAVED")
print("=" * 70)

print(OUTPUT_PATH)

print("\nSPEECH STEP 4 COMPLETE")
