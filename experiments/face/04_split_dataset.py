from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "face"
)

FEATURE_PATH = (
    OUTPUT_PATH
    / "face_hog_features.npy"
)

METADATA_PATH = (
    OUTPUT_PATH
    / "face_grouped_metadata.csv"
)


# =========================================================
# LOAD
# =========================================================

print("=" * 70)
print("STEP 4B - STRATIFIED GROUP SPLIT")
print("=" * 70)

X = np.load(FEATURE_PATH)

df = pd.read_csv(METADATA_PATH)

print("\nFeatures:", X.shape)
print("Metadata:", df.shape)


if len(X) != len(df):
    raise ValueError(
        "Feature rows and metadata rows do not match."
    )


# =========================================================
# TARGET / GROUP
# =========================================================

y = df["severity"].astype(str).to_numpy()

groups = (
    df["source_group"]
    .astype(str)
    .to_numpy()
)


# =========================================================
# FIRST SPLIT
#
# 5 folds = each fold ~20%
#
# Fold 0 -> TEST
# Remaining ~80% -> Train + Validation
# =========================================================

outer_split = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

outer_folds = list(
    outer_split.split(
        X,
        y,
        groups
    )
)

train_val_idx, test_idx = outer_folds[0]


# =========================================================
# SECOND SPLIT
#
# Need ~20% of TOTAL for validation.
#
# train_val currently ~80%.
# Splitting it into 4 folds means:
#
# 1/4 of 80% ≈ 20% total
#
# therefore:
# Train ≈ 60%
# Val   ≈ 20%
# Test  ≈ 20%
# =========================================================

X_train_val = X[train_val_idx]

df_train_val = (
    df.iloc[train_val_idx]
    .reset_index(drop=True)
)

y_train_val = (
    df_train_val["severity"]
    .astype(str)
    .to_numpy()
)

groups_train_val = (
    df_train_val["source_group"]
    .astype(str)
    .to_numpy()
)


inner_split = StratifiedGroupKFold(
    n_splits=4,
    shuffle=True,
    random_state=42
)

inner_folds = list(
    inner_split.split(
        X_train_val,
        y_train_val,
        groups_train_val
    )
)

train_inner_idx, val_inner_idx = (
    inner_folds[0]
)


# =========================================================
# FINAL DATASETS
# =========================================================

X_train = X_train_val[
    train_inner_idx
]

X_val = X_train_val[
    val_inner_idx
]

X_test = X[
    test_idx
]


df_train = (
    df_train_val
    .iloc[train_inner_idx]
    .reset_index(drop=True)
)

df_val = (
    df_train_val
    .iloc[val_inner_idx]
    .reset_index(drop=True)
)

df_test = (
    df.iloc[test_idx]
    .reset_index(drop=True)
)


# =========================================================
# LEAKAGE CHECK
# =========================================================

train_groups = set(
    df_train["source_group"]
)

val_groups = set(
    df_val["source_group"]
)

test_groups = set(
    df_test["source_group"]
)


train_val_overlap = (
    train_groups & val_groups
)

train_test_overlap = (
    train_groups & test_groups
)

val_test_overlap = (
    val_groups & test_groups
)


print("\n" + "=" * 70)
print("LEAKAGE CHECK")
print("=" * 70)

print(
    "Train <-> Validation:",
    len(train_val_overlap)
)

print(
    "Train <-> Test:",
    len(train_test_overlap)
)

print(
    "Validation <-> Test:",
    len(val_test_overlap)
)


if (
    train_val_overlap
    or train_test_overlap
    or val_test_overlap
):
    raise RuntimeError(
        "SOURCE GROUP LEAKAGE DETECTED"
    )


# =========================================================
# SUMMARY
# =========================================================

def show_summary(name, metadata):

    print("\n" + "-" * 60)

    print(name)

    print(
        "Images:",
        len(metadata)
    )

    print(
        "Source groups:",
        metadata[
            "source_group"
        ].nunique()
    )

    percentage = (
        len(metadata)
        / len(df)
        * 100
    )

    print(
        f"Percentage: {percentage:.2f}%"
    )

    print("\nSeverity:")

    counts = (
        metadata[
            "severity"
        ]
        .value_counts()
    )

    print(counts)

    print("\nSeverity percentage:")

    percentages = (
        metadata[
            "severity"
        ]
        .value_counts(
            normalize=True
        )
        .mul(100)
        .round(2)
    )

    print(percentages)

    print("\nRegion:")

    print(
        metadata[
            "region"
        ]
        .value_counts()
    )


print("\n" + "=" * 70)
print("SPLIT SUMMARY")
print("=" * 70)

show_summary(
    "TRAIN",
    df_train
)

show_summary(
    "VALIDATION",
    df_val
)

show_summary(
    "TEST",
    df_test
)


# =========================================================
# VERIFY ALL CLASSES EXIST
# =========================================================

expected_classes = set(
    df["severity"].unique()
)

for name, metadata in [
    ("TRAIN", df_train),
    ("VALIDATION", df_val),
    ("TEST", df_test)
]:

    classes = set(
        metadata[
            "severity"
        ].unique()
    )

    missing = (
        expected_classes
        - classes
    )

    if missing:

        raise RuntimeError(
            f"{name} missing classes: "
            f"{missing}"
        )


# =========================================================
# SAVE FEATURES
# =========================================================

np.save(
    OUTPUT_PATH
    / "X_train.npy",
    X_train
)

np.save(
    OUTPUT_PATH
    / "X_val.npy",
    X_val
)

np.save(
    OUTPUT_PATH
    / "X_test.npy",
    X_test
)


# =========================================================
# SAVE METADATA
# =========================================================

df_train.to_csv(
    OUTPUT_PATH
    / "train_metadata.csv",
    index=False,
    encoding="utf-8-sig"
)

df_val.to_csv(
    OUTPUT_PATH
    / "val_metadata.csv",
    index=False,
    encoding="utf-8-sig"
)

df_test.to_csv(
    OUTPUT_PATH
    / "test_metadata.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# SAVE SPLIT ASSIGNMENT
# =========================================================

train_save = df_train.copy()
train_save["split"] = "train"

val_save = df_val.copy()
val_save["split"] = "validation"

test_save = df_test.copy()
test_save["split"] = "test"

split_manifest = pd.concat(
    [
        train_save,
        val_save,
        test_save
    ],
    ignore_index=True
)

split_manifest.to_csv(
    OUTPUT_PATH
    / "face_split_manifest.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# FINAL
# =========================================================

print("\n" + "=" * 70)
print("STEP 4B COMPLETE")
print("=" * 70)

print(
    "\nAll source groups are isolated "
    "between Train / Validation / Test."
)

print(
    "\nSaved:"
)

print(
    OUTPUT_PATH
    / "face_split_manifest.csv"
)