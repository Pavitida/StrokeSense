from pathlib import Path
from io import BytesIO

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "torgo"
    / "data"
)

SPLIT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "speech"
    / "torgo_split.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "speech"
    / "torgo_mfcc_features.csv"
)


# ============================================================
# SETTINGS
# ============================================================

TARGET_SR = 16000
N_MFCC = 13

print("=" * 70)
print("SPEECH STEP 5 - EXTRACT MFCC FEATURES")
print("=" * 70)

print("\nTarget sample rate:", TARGET_SR)
print("MFCC coefficients:", N_MFCC)


# ============================================================
# LOAD SPLIT METADATA
# ============================================================

metadata = pd.read_csv(SPLIT_PATH)

print("\nSamples:")
print(len(metadata))

print("\nSpeakers:")
print(metadata["speaker_id"].nunique())

print("\nSplit:")
print(metadata["split"].value_counts())


# ============================================================
# LOOKUP
# ============================================================

# Key = (shard filename, row index)
metadata_lookup = {
    (row["shard"], int(row["row_index"])): row
    for _, row in metadata.iterrows()
}


# ============================================================
# FEATURE EXTRACTION
# ============================================================

results = []

success = 0
failed = 0

parquet_files = sorted(
    DATA_DIR.glob("train-*.parquet")
)

for parquet_file in parquet_files:

    print("\n" + "=" * 70)
    print("Processing:", parquet_file.name)
    print("=" * 70)

    df = pd.read_parquet(parquet_file)

    for row_index, row in df.iterrows():

        key = (
            parquet_file.name,
            int(row_index)
        )

        if key not in metadata_lookup:
            continue

        meta = metadata_lookup[key]

        try:

            # ----------------------------------------------
            # READ AUDIO BYTES
            # ----------------------------------------------

            audio_info = row["audio"]

            audio_bytes = audio_info["bytes"]

            y, sr = sf.read(
                BytesIO(audio_bytes),
                dtype="float32"
            )

            # Stereo -> mono
            if y.ndim > 1:
                y = np.mean(y, axis=1)

            # ----------------------------------------------
            # RESAMPLE
            # ----------------------------------------------

            if sr != TARGET_SR:

                y = librosa.resample(
                    y,
                    orig_sr=sr,
                    target_sr=TARGET_SR
                )

                sr = TARGET_SR

            # ----------------------------------------------
            # BASIC CHECK
            # ----------------------------------------------

            if len(y) == 0:
                raise ValueError("Empty audio")

            # ----------------------------------------------
            # NORMALIZE AMPLITUDE
            # ----------------------------------------------

            peak = np.max(np.abs(y))

            if peak > 0:
                y = y / peak

            # ----------------------------------------------
            # MFCC
            # ----------------------------------------------

            mfcc = librosa.feature.mfcc(
                y=y,
                sr=sr,
                n_mfcc=N_MFCC
            )

            # Each recording has different duration.
            # Summarize each coefficient using mean + std.
            mfcc_mean = np.mean(
                mfcc,
                axis=1
            )

            mfcc_std = np.std(
                mfcc,
                axis=1
            )

            feature_row = {
                "speaker_id": meta["speaker_id"],
                "speech_status": meta["speech_status"],
                "gender": meta["gender"],
                "split": meta["split"],
                "shard": parquet_file.name,
                "row_index": int(row_index),
                "audio_path": meta["audio_path"],
                "duration": meta["duration"],
            }

            for i in range(N_MFCC):

                feature_row[
                    f"mfcc_{i + 1}_mean"
                ] = mfcc_mean[i]

                feature_row[
                    f"mfcc_{i + 1}_std"
                ] = mfcc_std[i]

            results.append(feature_row)

            success += 1

        except Exception as e:

            failed += 1

            print(
                f"\nFAILED "
                f"{parquet_file.name} "
                f"row {row_index}: {e}"
            )

        total_processed = success + failed

        if total_processed % 500 == 0:

            print(
                f"Processed "
                f"{total_processed}/"
                f"{len(metadata)}"
            )


# ============================================================
# DATAFRAME
# ============================================================

features = pd.DataFrame(results)


# ============================================================
# CHECK
# ============================================================

print("\n" + "=" * 70)
print("MFCC RESULT")
print("=" * 70)

print("\nSuccessful:")
print(success)

print("\nFailed:")
print(failed)

print("\nFeature rows:")
print(len(features))

print("\nSpeakers:")
print(features["speaker_id"].nunique())

print("\nSplit:")
print(features["split"].value_counts())

print("\nLabels:")
print(features["speech_status"].value_counts())


feature_columns = [
    c
    for c in features.columns
    if c.startswith("mfcc_")
]

print("\nMFCC feature count:")
print(len(feature_columns))

print("\nNaN values:")
print(
    features[
        feature_columns
    ].isna().sum().sum()
)

print("\nInfinite values:")
print(
    np.isinf(
        features[
            feature_columns
        ].values
    ).sum()
)


# ============================================================
# SAFETY CHECKS
# ============================================================

assert len(feature_columns) == 26, (
    f"Expected 26 MFCC features, "
    f"got {len(feature_columns)}"
)

assert (
    features["speaker_id"]
    .isna()
    .sum()
    == 0
), "Missing speaker IDs"

speaker_split_count = (
    features
    .groupby("speaker_id")["split"]
    .nunique()
)

assert (
    speaker_split_count.max() == 1
), "Speaker leakage detected"


# ============================================================
# SAVE
# ============================================================

features.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 70)
print("SAVED")
print("=" * 70)

print(OUTPUT_PATH)

print("\nSPEECH STEP 5 COMPLETE")