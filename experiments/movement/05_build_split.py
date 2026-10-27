from pathlib import Path
from itertools import combinations
import pandas as pd
import numpy as np

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "movement"
    / "movement_metadata.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "movement"
    / "movement_split.csv"
)

# ============================================================
# LOAD
# ============================================================

print("\n" + "=" * 70)
print("MOVEMENT STEP 5 - PERSON-AWARE SPLIT")
print("=" * 70)

df = pd.read_csv(INPUT_PATH)

print("\nTotal videos:")
print(len(df))

print("\nTotal people:")
print(df["id_part1"].nunique())

# Force ID to 2-digit string
df["id_part1"] = (
    df["id_part1"]
    .astype(str)
    .str.zfill(2)
)

# ============================================================
# ORIGINAL TRAIN / TEST PEOPLE
# ============================================================

train_pool = sorted(
    df.loc[
        df["original_split"].str.lower() == "train",
        "id_part1"
    ].unique()
)

test_people = sorted(
    df.loc[
        df["original_split"].str.lower() == "test",
        "id_part1"
    ].unique()
)

print("\nOriginal train people:")
print(train_pool)

print("\nOriginal test people:")
print(test_people)

# ============================================================
# FIND BEST VALIDATION PAIR
# ============================================================

# We keep original test completely untouched.
# From original training subjects, choose 2 people for validation.
#
# Score is based on:
# 1. Complete / Incomplete balance
# 2. Exercise distribution
# 3. Validation size

candidate_results = []

all_exercises = sorted(df["exercise"].unique())

for val_pair in combinations(train_pool, 2):

    val_pair = list(val_pair)

    val_df = df[df["id_part1"].isin(val_pair)]

    # --------------------------------------------------------
    # Label balance
    # --------------------------------------------------------

    label_counts = (
        val_df["label"]
        .value_counts(normalize=True)
    )

    complete_ratio = label_counts.get("Complete", 0)
    incomplete_ratio = label_counts.get("Incomplete", 0)

    label_balance_error = abs(
        complete_ratio - incomplete_ratio
    )

    # --------------------------------------------------------
    # Exercise coverage / balance
    # --------------------------------------------------------

    exercise_counts = (
        val_df["exercise"]
        .value_counts()
        .reindex(all_exercises, fill_value=0)
    )

    missing_exercises = int(
        (exercise_counts == 0).sum()
    )

    if exercise_counts.sum() > 0:
        exercise_ratio = (
            exercise_counts / exercise_counts.sum()
        )

        ideal_ratio = 1 / len(all_exercises)

        exercise_balance_error = float(
            np.abs(
                exercise_ratio - ideal_ratio
            ).sum()
        )
    else:
        exercise_balance_error = 999

    # --------------------------------------------------------
    # Validation size
    # --------------------------------------------------------

    val_size = len(val_df)

    target_size = len(
        df[
            df["id_part1"].isin(train_pool)
        ]
    ) * (2 / len(train_pool))

    size_error = abs(
        val_size - target_size
    ) / target_size

    # --------------------------------------------------------
    # TOTAL SCORE
    # Lower = better
    # --------------------------------------------------------

    score = (
        label_balance_error
        + exercise_balance_error
        + size_error * 0.25
        + missing_exercises * 10
    )

    candidate_results.append({
        "validation_people": val_pair,
        "videos": val_size,
        "complete": int(
            (val_df["label"] == "Complete").sum()
        ),
        "incomplete": int(
            (val_df["label"] == "Incomplete").sum()
        ),
        "missing_exercises": missing_exercises,
        "score": score,
    })

# ============================================================
# SORT CANDIDATES
# ============================================================

candidate_results = sorted(
    candidate_results,
    key=lambda x: x["score"]
)

print("\n" + "=" * 70)
print("TOP VALIDATION CANDIDATES")
print("=" * 70)

for result in candidate_results[:5]:

    print(
        f"\nPeople: {result['validation_people']}"
    )

    print(
        f"Videos: {result['videos']}"
    )

    print(
        f"Complete: {result['complete']}"
    )

    print(
        f"Incomplete: {result['incomplete']}"
    )

    print(
        f"Missing exercises: "
        f"{result['missing_exercises']}"
    )

    print(
        f"Score: {result['score']:.4f}"
    )

# ============================================================
# BEST VALIDATION PAIR
# ============================================================

validation_people = (
    candidate_results[0]["validation_people"]
)

train_people = [
    p for p in train_pool
    if p not in validation_people
]

print("\n" + "=" * 70)
print("SELECTED SPLIT")
print("=" * 70)

print("\nTrain people:")
print(train_people)

print("\nValidation people:")
print(validation_people)

print("\nTest people:")
print(test_people)

# ============================================================
# ASSIGN SPLIT
# ============================================================

def assign_split(person_id):

    if person_id in train_people:
        return "train"

    if person_id in validation_people:
        return "validation"

    if person_id in test_people:
        return "test"

    return "unknown"


df["split"] = df["id_part1"].apply(
    assign_split
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

    subset = df[
        df["split"] == split_name
    ]

    print("\n" + "-" * 70)
    print(split_name.upper())
    print("-" * 70)

    print("\nVideos:")
    print(len(subset))

    print("\nPeople:")
    print(subset["id_part1"].nunique())

    print("\nPerson IDs:")
    print(
        sorted(
            subset["id_part1"].unique()
        )
    )

    print("\nLabels:")
    print(
        subset["label"].value_counts()
    )

    print("\nExercises:")
    print(
        subset["exercise"].value_counts()
    )

    print("\nExercise x Label:")
    print(
        pd.crosstab(
            subset["exercise"],
            subset["label"]
        )
    )

# ============================================================
# LEAKAGE CHECK
# ============================================================

train_set = set(
    df.loc[
        df["split"] == "train",
        "id_part1"
    ]
)

val_set = set(
    df.loc[
        df["split"] == "validation",
        "id_part1"
    ]
)

test_set = set(
    df.loc[
        df["split"] == "test",
        "id_part1"
    ]
)

print("\n" + "=" * 70)
print("LEAKAGE CHECK")
print("=" * 70)

print(
    "\nTrain / Validation overlap:",
    train_set & val_set
)

print(
    "Train / Test overlap:",
    train_set & test_set
)

print(
    "Validation / Test overlap:",
    val_set & test_set
)

assert len(train_set & val_set) == 0
assert len(train_set & test_set) == 0
assert len(val_set & test_set) == 0

print("\nPASS - no person leakage.")

# ============================================================
# UNKNOWN CHECK
# ============================================================

unknown_count = (
    df["split"] == "unknown"
).sum()

print("\nUnknown split videos:")
print(unknown_count)

assert unknown_count == 0

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

print("\n" + str(OUTPUT_PATH))

print(
    "\nMOVEMENT STEP 5 COMPLETE"
)