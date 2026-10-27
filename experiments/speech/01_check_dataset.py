from pathlib import Path
import pandas as pd

# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "torgo"
    / "data"
    / "train-00000-of-00004.parquet"
)

print("=" * 70)
print("SPEECH STEP 1 - CHECK TORGO DATASET")
print("=" * 70)

print("\nDataset:")
print(DATA_PATH)

print("\nExists:")
print(DATA_PATH.exists())

if not DATA_PATH.exists():
    raise FileNotFoundError(DATA_PATH)

# ============================================================
# LOAD
# ============================================================

print("\nLoading parquet...")

df = pd.read_parquet(DATA_PATH)

print("\n" + "=" * 70)
print("BASIC INFORMATION")
print("=" * 70)

print("\nRows:")
print(len(df))

print("\nColumns:")
for col in df.columns:
    print("•", col)

print("\nShape:")
print(df.shape)

print("\nData types:")
print(df.dtypes)

# ============================================================
# PREVIEW
# ============================================================

print("\n" + "=" * 70)
print("PREVIEW")
print("=" * 70)

print(df.head(5).to_string())

# ============================================================
# MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

print(df.isna().sum())

# ============================================================
# SPEECH STATUS
# ============================================================

if "speech_status" in df.columns:

    print("\n" + "=" * 70)
    print("SPEECH STATUS")
    print("=" * 70)

    print(df["speech_status"].value_counts(dropna=False))

# ============================================================
# GENDER
# ============================================================

if "gender" in df.columns:

    print("\n" + "=" * 70)
    print("GENDER")
    print("=" * 70)

    print(df["gender"].value_counts(dropna=False))

# ============================================================
# AUDIO
# ============================================================

if "audio" in df.columns:

    print("\n" + "=" * 70)
    print("AUDIO COLUMN")
    print("=" * 70)

    first_audio = df.iloc[0]["audio"]

    print("\nType:")
    print(type(first_audio))

    print("\nFirst audio value:")
    print(first_audio)

# ============================================================
# TRANSCRIPTION
# ============================================================

if "transcription" in df.columns:

    print("\n" + "=" * 70)
    print("TRANSCRIPTION EXAMPLES")
    print("=" * 70)

    for text in df["transcription"].head(10):
        print("•", text)

# ============================================================
# POSSIBLE SPEAKER INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("POSSIBLE SPEAKER COLUMNS")
print("=" * 70)

speaker_keywords = [
    "speaker",
    "subject",
    "patient",
    "user",
    "id",
    "file",
    "path",
]

possible_columns = []

for col in df.columns:

    col_lower = col.lower()

    if any(
        keyword in col_lower
        for keyword in speaker_keywords
    ):
        possible_columns.append(col)

if possible_columns:

    for col in possible_columns:

        print(f"\nCOLUMN: {col}")

        print(
            df[col]
            .astype(str)
            .head(20)
            .to_string(index=False)
        )

else:

    print("\nNo obvious speaker column found.")

# ============================================================
# UNIQUE VALUES
# ============================================================

print("\n" + "=" * 70)
print("LOW-CARDINALITY COLUMNS")
print("=" * 70)

for col in df.columns:

    try:
        n_unique = df[col].nunique()

        if n_unique <= 30:

            print(
                f"\n{col}: "
                f"{n_unique} unique values"
            )

            print(
                df[col]
                .value_counts(dropna=False)
                .head(30)
            )

    except TypeError:
        pass

print("\n" + "=" * 70)
print("SPEECH STEP 1 COMPLETE")
print("=" * 70)