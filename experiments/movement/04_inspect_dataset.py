from pathlib import Path
import pandas as pd
from collections import Counter

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "movement"
    / "An upper limb stroke rehabilitation exercise video"
    / "Exercise"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "movement"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "movement_metadata.csv"


print("=" * 70)
print("MOVEMENT STEP 4 - INSPECT DATASET")
print("=" * 70)

print("\nDataset:")
print(DATA_ROOT)

print("\nExists:")
print(DATA_ROOT.exists())

if not DATA_ROOT.exists():
    raise FileNotFoundError(DATA_ROOT)


# ============================================================
# FIND VIDEOS
# ============================================================

videos = sorted(DATA_ROOT.rglob("*.mp4"))

print("\nTotal videos:")
print(len(videos))

if len(videos) == 0:
    raise RuntimeError("No MP4 videos found.")


# ============================================================
# PARSE METADATA
# ============================================================

rows = []

for video in videos:

    relative = video.relative_to(DATA_ROOT)

    parts = relative.parts

    # Expected:
    #
    # exercise /
    # class /
    # original_split /
    # filename.mp4

    if len(parts) != 4:
        print("Unexpected path:")
        print(relative)
        continue

    exercise = parts[0]
    label = parts[1]
    original_split = parts[2]
    filename = parts[3]

    stem = video.stem

    filename_parts = stem.split("_")

    if len(filename_parts) != 4:
        print("Unexpected filename:")
        print(filename)
        continue

    part1, part2, part3, part4 = filename_parts

    rows.append(
        {
            "path": str(video),
            "exercise": exercise,
            "label": label,
            "original_split": original_split,
            "filename": filename,

            # Do NOT assume meanings yet
            "id_part1": part1,
            "id_part2": part2,
            "id_part3": part3,
            "id_part4": part4,
        }
    )


df = pd.DataFrame(rows)


# ============================================================
# BASIC SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print("\nParsed videos:")
print(len(df))

print("\nExercises:")
print(df["exercise"].value_counts().sort_index())

print("\nLabels:")
print(df["label"].value_counts())

print("\nOriginal split:")
print(df["original_split"].value_counts())


# ============================================================
# EXERCISE x LABEL
# ============================================================

print("\n" + "=" * 70)
print("EXERCISE x LABEL")
print("=" * 70)

table = pd.crosstab(
    df["exercise"],
    df["label"]
)

print(table)


# ============================================================
# FILENAME COMPONENTS
# ============================================================

print("\n" + "=" * 70)
print("FILENAME COMPONENT ANALYSIS")
print("=" * 70)

for column in [
    "id_part1",
    "id_part2",
    "id_part3",
    "id_part4",
]:

    print(f"\n{column}")
    print("-" * 40)

    values = sorted(df[column].unique())

    print("Unique count:")
    print(len(values))

    print("Values:")
    print(values)


# ============================================================
# PART 1 ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("ID PART 1 ANALYSIS")
print("=" * 70)

print("\nVideos by id_part1:")

print(
    df["id_part1"]
    .value_counts()
    .sort_index()
)


print("\nID part1 x original split:")

print(
    pd.crosstab(
        df["id_part1"],
        df["original_split"]
    )
)


print("\nID part1 x label:")

print(
    pd.crosstab(
        df["id_part1"],
        df["label"]
    )
)


print("\nID part1 x exercise:")

print(
    pd.crosstab(
        df["id_part1"],
        df["exercise"]
    )
)


# ============================================================
# CHECK SPLIT OVERLAP
# ============================================================

print("\n" + "=" * 70)
print("ORIGINAL SPLIT OVERLAP CHECK")
print("=" * 70)

train_ids = set(
    df.loc[
        df["original_split"].str.lower() == "train",
        "id_part1"
    ]
)

test_ids = set(
    df.loc[
        df["original_split"].str.lower() == "test",
        "id_part1"
    ]
)

overlap = train_ids & test_ids

print("\nTrain id_part1:")
print(sorted(train_ids))

print("\nTest id_part1:")
print(sorted(test_ids))

print("\nOverlap:")
print(sorted(overlap))

print("\nOverlap count:")
print(len(overlap))


# ============================================================
# SAMPLE FILENAMES PER ID
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE FILES BY ID PART 1")
print("=" * 70)

for subject_id in sorted(df["id_part1"].unique()):

    subset = df[df["id_part1"] == subject_id]

    print(f"\nID {subject_id}")

    for name in subset["filename"].head(5):
        print(" •", name)


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

print("\nMOVEMENT STEP 4 COMPLETE")