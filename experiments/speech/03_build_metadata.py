from pathlib import Path
import pandas as pd
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "torgo" / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "speech"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "torgo_metadata.csv"

files = sorted(DATA_DIR.glob("train-*.parquet"))

print("=" * 70)
print("SPEECH STEP 3 - BUILD TORGO METADATA")
print("=" * 70)

rows = []

for shard_index, file in enumerate(files):

    print(f"\nReading {file.name}...")

    df = pd.read_parquet(file)

    for row_index, row in df.iterrows():

        audio = row["audio"]

        audio_path = audio.get("path", "") if isinstance(audio, dict) else ""

        # Examples:
        # FC01_1_arrayMic_0066.wav -> FC01
        # M02_...wav              -> M02
        match = re.match(
            r"^([A-Za-z]+\d+)",
            Path(audio_path).name
        )

        speaker_id = (
            match.group(1).upper()
            if match
            else None
        )

        rows.append({
            "shard": file.name,
            "row_index": row_index,
            "audio_path": audio_path,
            "speaker_id": speaker_id,
            "transcription": row["transcription"],
            "speech_status": row["speech_status"],
            "gender": row["gender"],
            "duration": row["duration"],
        })

metadata = pd.DataFrame(rows)

print("\n" + "=" * 70)
print("DATASET")
print("=" * 70)

print("\nTotal samples:")
print(len(metadata))

print("\nSpeech status:")
print(metadata["speech_status"].value_counts())

print("\nGender:")
print(metadata["gender"].value_counts())

print("\nMissing speaker IDs:")
print(metadata["speaker_id"].isna().sum())

print("\nUnique speakers:")
print(metadata["speaker_id"].nunique())

print("\nSpeaker IDs:")
print(
    sorted(
        metadata["speaker_id"]
        .dropna()
        .unique()
        .tolist()
    )
)

# ============================================================
# SPEAKER SUMMARY
# ============================================================

speaker_summary = (
    metadata
    .groupby("speaker_id")
    .agg(
        samples=("audio_path", "size"),
        speech_status=("speech_status", lambda x: ", ".join(sorted(set(x)))),
        gender=("gender", lambda x: ", ".join(sorted(set(x)))),
        total_duration=("duration", "sum"),
    )
    .reset_index()
)

print("\n" + "=" * 70)
print("SPEAKER SUMMARY")
print("=" * 70)

print(
    speaker_summary
    .to_string(index=False)
)

# ============================================================
# CHECK LABEL CONSISTENCY
# ============================================================

label_counts = (
    metadata
    .groupby("speaker_id")["speech_status"]
    .nunique()
)

inconsistent = label_counts[
    label_counts > 1
]

print("\n" + "=" * 70)
print("SPEAKER / LABEL CONSISTENCY")
print("=" * 70)

print("\nSpeakers appearing in multiple labels:")
print(len(inconsistent))

if len(inconsistent) > 0:

    print("\nWARNING:")
    print(inconsistent)

else:

    print("PASS - every speaker has only one speech-status label.")

# ============================================================
# CLASS BY SPEAKER
# ============================================================

speaker_labels = (
    metadata[
        ["speaker_id", "speech_status"]
    ]
    .drop_duplicates()
)

print("\n" + "=" * 70)
print("SPEAKERS BY CLASS")
print("=" * 70)

print(
    speaker_labels[
        "speech_status"
    ].value_counts()
)

# ============================================================
# SAVE
# ============================================================

metadata.to_csv(
    OUTPUT_PATH,
    index=False
)

speaker_summary.to_csv(
    OUTPUT_DIR / "torgo_speaker_summary.csv",
    index=False
)

print("\n" + "=" * 70)
print("SAVED")
print("=" * 70)

print(OUTPUT_PATH)

print(
    OUTPUT_DIR /
    "torgo_speaker_summary.csv"
)

print("\nSPEECH STEP 3 COMPLETE")