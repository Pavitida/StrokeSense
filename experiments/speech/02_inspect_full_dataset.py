from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "torgo" / "data"

files = sorted(DATA_DIR.glob("train-*.parquet"))

print("=" * 70)
print("SPEECH STEP 2 - INSPECT FULL TORGO DATASET")
print("=" * 70)

print("\nParquet files:", len(files))

if len(files) != 4:
    raise RuntimeError(f"Expected 4 parquet files, found {len(files)}")


all_summary = []

for file in files:

    print("\n" + "=" * 70)
    print(file.name)
    print("=" * 70)

    # Read metadata columns first.
    # We intentionally avoid printing the audio bytes.
    df = pd.read_parquet(file)

    print("\nRows:")
    print(len(df))

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nSpeech status:")
    if "speech_status" in df.columns:
        print(df["speech_status"].value_counts(dropna=False))
    else:
        print("NO speech_status COLUMN")

    print("\nGender:")
    if "gender" in df.columns:
        print(df["gender"].value_counts(dropna=False))
    else:
        print("NO gender COLUMN")

    # Show everything except the large audio object
    display_columns = [
        c for c in df.columns
        if c != "audio"
    ]

    print("\nFirst 5 rows (audio excluded):")
    print(
        df[display_columns]
        .head()
        .to_string(index=False)
    )

    summary = {
        "file": file.name,
        "rows": len(df)
    }

    if "speech_status" in df.columns:

        counts = (
            df["speech_status"]
            .astype(str)
            .value_counts()
            .to_dict()
        )

        for label, count in counts.items():
            summary[f"status_{label}"] = count

    if "gender" in df.columns:

        counts = (
            df["gender"]
            .astype(str)
            .value_counts()
            .to_dict()
        )

        for gender, count in counts.items():
            summary[f"gender_{gender}"] = count

    all_summary.append(summary)

    del df


# ============================================================
# SUMMARY
# ============================================================

summary_df = pd.DataFrame(all_summary).fillna(0)

print("\n" + "=" * 70)
print("SHARD SUMMARY")
print("=" * 70)

print(summary_df.to_string(index=False))


print("\n" + "=" * 70)
print("TOTAL")
print("=" * 70)

print("\nTotal rows:")
print(int(summary_df["rows"].sum()))


# ============================================================
# PARQUET SCHEMA
# ============================================================

print("\n" + "=" * 70)
print("SCHEMA")
print("=" * 70)

sample = pd.read_parquet(files[0])

for column in sample.columns:

    print(
        f"{column}: {sample[column].dtype}"
    )


# ============================================================
# AUDIO STRUCTURE
# ============================================================

if "audio" in sample.columns:

    print("\n" + "=" * 70)
    print("AUDIO STRUCTURE")
    print("=" * 70)

    audio = sample.iloc[0]["audio"]

    print("\nPython type:")
    print(type(audio))

    if isinstance(audio, dict):

        print("\nAudio keys:")
        print(audio.keys())

        for key, value in audio.items():

            print(
                f"\n{key}: "
                f"type={type(value)}"
            )

            # Do not print huge byte arrays
            if isinstance(value, (bytes, bytearray)):
                print(
                    f"byte length={len(value)}"
                )
            else:
                text = str(value)

                if len(text) > 300:
                    text = text[:300] + "..."

                print(text)


print("\n" + "=" * 70)
print("STEP 2 COMPLETE")
print("=" * 70)