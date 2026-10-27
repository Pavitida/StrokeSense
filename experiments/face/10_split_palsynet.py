from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "face_landmark"
    / "palsynet_asymmetry_features.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "face_landmark"
    / "palsynet_split.csv"
)


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("STEP 10 - PERSON-AWARE DATA SPLIT")
print("=" * 70)

df = pd.read_csv(INPUT_PATH)

print("\nTotal frames:")
print(len(df))

print("\nTotal people:")
print(df["person_id"].nunique())


# ============================================================
# CREATE PERSON TABLE
# ============================================================

people = (
    df[
        ["person_id", "label"]
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)


# Safety check:
# each person must have only one label

label_check = (
    people.groupby("person_id")["label"]
    .nunique()
)

bad_people = label_check[
    label_check > 1
]

if len(bad_people) > 0:

    print("\nERROR:")
    print(
        "Some people appear in multiple labels."
    )

    print(bad_people)

    raise RuntimeError(
        "Person/label inconsistency detected."
    )


print("\nPeople by label:")

print(
    people["label"].value_counts()
)


# ============================================================
# SPLIT PEOPLE
#
# approximately:
#
# Train = 60%
# Validation = 20%
# Test = 20%
#
# ============================================================

train_people, temp_people = train_test_split(

    people,

    test_size=0.40,

    random_state=42,

    stratify=people["label"]
)


val_people, test_people = train_test_split(

    temp_people,

    test_size=0.50,

    random_state=42,

    stratify=temp_people["label"]
)


# ============================================================
# CREATE MAPPING
# ============================================================

split_map = {}


for person in train_people["person_id"]:
    split_map[person] = "train"


for person in val_people["person_id"]:
    split_map[person] = "validation"


for person in test_people["person_id"]:
    split_map[person] = "test"


df["split"] = (
    df["person_id"]
    .map(split_map)
)


# ============================================================
# SAFETY CHECK
# ============================================================

print("\n" + "=" * 70)
print("LEAKAGE CHECK")
print("=" * 70)


train_ids = set(
    df[
        df["split"] == "train"
    ]["person_id"]
)


val_ids = set(
    df[
        df["split"] == "validation"
    ]["person_id"]
)


test_ids = set(
    df[
        df["split"] == "test"
    ]["person_id"]
)


train_val_overlap = (
    train_ids & val_ids
)

train_test_overlap = (
    train_ids & test_ids
)

val_test_overlap = (
    val_ids & test_ids
)


print(
    "Train / Validation overlap:",
    len(train_val_overlap)
)

print(
    "Train / Test overlap:",
    len(train_test_overlap)
)

print(
    "Validation / Test overlap:",
    len(val_test_overlap)
)


if (
    len(train_val_overlap) > 0
    or len(train_test_overlap) > 0
    or len(val_test_overlap) > 0
):

    raise RuntimeError(
        "DATA LEAKAGE DETECTED"
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

    print(
        f"\n{split_name.upper()}"
    )

    print(
        "Frames:",
        len(part)
    )

    print(
        "People:",
        part["person_id"].nunique()
    )

    print("\nPeople by label:")

    person_counts = (
        part[
            ["person_id", "label"]
        ]
        .drop_duplicates()
        ["label"]
        .value_counts()
    )

    print(
        person_counts
    )

    print("\nFrames by label:")

    print(
        part["label"].value_counts()
    )


# ============================================================
# SAVE
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\n" + "=" * 70)
print("SAVED")
print("=" * 70)

print(OUTPUT_PATH)

print("\nSTEP 10 COMPLETE")