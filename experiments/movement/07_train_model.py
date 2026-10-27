from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "movement"
    / "movement_pose_features.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "movement"
)

MODEL_PATH = (
    MODEL_DIR
    / "movement_model.pkl"
)

VALIDATION_OUTPUT_PATH = (
    OUTPUT_DIR
    / "movement_validation_predictions.csv"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

LABEL_MAP = {
    "Complete": 0,
    "Incomplete": 1,
}

THRESHOLD = 0.5


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("MOVEMENT STEP 7 - TRAIN MOVEMENT MODEL")
print("=" * 70)

if not FEATURE_PATH.exists():
    raise FileNotFoundError(
        f"Feature file not found:\n{FEATURE_PATH}"
    )

df = pd.read_csv(FEATURE_PATH)

print("\nFeature file:")
print(FEATURE_PATH)

print("\nTotal samples:")
print(len(df))

print("\nPeople:")
print(df["person_id"].nunique())

print("\nSplit:")
print(df["split"].value_counts())

print("\nLabels:")
print(df["label"].value_counts())

print("\nExercises:")
print(df["exercise"].value_counts())


# ============================================================
# CHECK LABELS
# ============================================================

unknown_labels = (
    set(df["label"].dropna().unique())
    - set(LABEL_MAP.keys())
)

if unknown_labels:
    raise ValueError(
        f"Unknown labels found: {unknown_labels}"
    )

df["target"] = df["label"].map(LABEL_MAP)


# ============================================================
# SELECT FEATURES
# ============================================================

# These columns must NOT be used for training.
# They are metadata / video quality / identifiers rather than
# movement biomechanics.

EXCLUDE_COLUMNS = {
    # identifiers / metadata
    "video_path",
    "filename",
    "person_id",
    "exercise",
    "label",
    "split",
    "original_split",
    "target",

    # video properties
    "video_fps",
    "video_total_frames",
    "processed_frames",
    "detected_frames",

    # pose quality
    "pose_detection_rate",
}


numeric_columns = (
    df.select_dtypes(include=[np.number])
    .columns
    .tolist()
)

feature_columns = [
    col
    for col in numeric_columns
    if col not in EXCLUDE_COLUMNS
]


print("\n" + "=" * 70)
print("FEATURE SELECTION")
print("=" * 70)

print("\nNumeric columns:")
print(len(numeric_columns))

print("\nSelected movement features:")
print(len(feature_columns))

for feature in feature_columns:
    print("•", feature)


if len(feature_columns) == 0:
    raise RuntimeError(
        "No movement features were selected."
    )


# ============================================================
# CHECK FEATURES
# ============================================================

X_all = df[feature_columns].copy()

nan_count = int(
    X_all.isna().sum().sum()
)

inf_count = int(
    np.isinf(
        X_all.to_numpy(dtype=float)
    ).sum()
)

print("\nNaN values:")
print(nan_count)

print("\nInfinite values:")
print(inf_count)

if nan_count > 0:
    raise ValueError(
        "NaN values detected in movement features."
    )

if inf_count > 0:
    raise ValueError(
        "Infinite values detected in movement features."
    )


# ============================================================
# TRAIN / VALIDATION
# ============================================================

train_df = (
    df[df["split"] == "train"]
    .copy()
)

validation_df = (
    df[df["split"] == "validation"]
    .copy()
)


if len(train_df) == 0:
    raise RuntimeError(
        "No training samples found."
    )

if len(validation_df) == 0:
    raise RuntimeError(
        "No validation samples found."
    )


X_train = train_df[feature_columns]
y_train = train_df["target"]

X_val = validation_df[feature_columns]
y_val = validation_df["target"]


# ============================================================
# DATA SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DATA")
print("=" * 70)

print("\nTrain:")
print("Samples:", len(train_df))
print(
    "People:",
    train_df["person_id"].nunique()
)
print(
    "Person IDs:",
    sorted(
        train_df["person_id"]
        .astype(str)
        .unique()
        .tolist()
    )
)

print("\nLabels:")
print(
    train_df["label"]
    .value_counts()
)


print("\nValidation:")
print("Samples:", len(validation_df))
print(
    "People:",
    validation_df["person_id"].nunique()
)
print(
    "Person IDs:",
    sorted(
        validation_df["person_id"]
        .astype(str)
        .unique()
        .tolist()
    )
)

print("\nLabels:")
print(
    validation_df["label"]
    .value_counts()
)


# ============================================================
# LEAKAGE CHECK
# ============================================================

train_people = set(
    train_df["person_id"]
    .astype(str)
)

validation_people = set(
    validation_df["person_id"]
    .astype(str)
)

overlap = (
    train_people
    & validation_people
)

print("\n" + "=" * 70)
print("LEAKAGE CHECK")
print("=" * 70)

print(
    "\nTrain / Validation person overlap:",
    overlap
)

if overlap:
    raise RuntimeError(
        "Person leakage detected between "
        "train and validation."
    )

print(
    "\nPASS - no person leakage."
)


# ============================================================
# MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 70)


model = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
    ]
)


model.fit(
    X_train,
    y_train
)


# ============================================================
# VALIDATION PREDICTION
# ============================================================

validation_probability = (
    model.predict_proba(
        X_val
    )[:, 1]
)

validation_prediction = (
    validation_probability
    >= THRESHOLD
).astype(int)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_val,
    validation_prediction
)

precision = precision_score(
    y_val,
    validation_prediction,
    zero_division=0
)

recall = recall_score(
    y_val,
    validation_prediction,
    zero_division=0
)

f1 = f1_score(
    y_val,
    validation_prediction,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_val,
    validation_probability
)


print("\n" + "=" * 70)
print("VIDEO-LEVEL VALIDATION")
print("=" * 70)

print(
    f"\nAccuracy: {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall: {recall:.4f}"
)

print(
    f"F1-score: {f1:.4f}"
)

print(
    f"ROC-AUC: {roc_auc:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification report:\n")

print(
    classification_report(
        y_val,
        validation_prediction,
        target_names=[
            "Complete",
            "Incomplete",
        ],
        digits=4,
        zero_division=0,
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_val,
    validation_prediction
)

print("Confusion matrix:\n")

print(cm)


# ============================================================
# PER-EXERCISE VALIDATION
# ============================================================

validation_results = (
    validation_df[
        [
            "person_id",
            "exercise",
            "label",
            "split",
        ]
    ]
    .copy()
)

validation_results[
    "incomplete_probability"
] = validation_probability

validation_results[
    "prediction"
] = validation_prediction

validation_results[
    "predicted_label"
] = validation_results[
    "prediction"
].map(
    {
        0: "Complete",
        1: "Incomplete",
    }
)


print("\n" + "=" * 70)
print("VALIDATION BY EXERCISE")
print("=" * 70)


exercise_results = []

for exercise in sorted(
    validation_results[
        "exercise"
    ].unique()
):

    mask = (
        validation_results[
            "exercise"
        ]
        == exercise
    )

    subset = (
        validation_results[
            mask
        ]
    )

    y_true = (
        validation_df.loc[
            subset.index,
            "target"
        ]
    )

    y_pred = (
        subset[
            "prediction"
        ]
    )

    y_prob = (
        subset[
            "incomplete_probability"
        ]
    )

    ex_accuracy = accuracy_score(
        y_true,
        y_pred
    )

    ex_precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    ex_recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    ex_f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    if y_true.nunique() == 2:
        ex_auc = roc_auc_score(
            y_true,
            y_prob
        )
    else:
        ex_auc = np.nan

    exercise_results.append(
        {
            "exercise": exercise,
            "samples": len(subset),
            "accuracy": ex_accuracy,
            "precision": ex_precision,
            "recall": ex_recall,
            "f1": ex_f1,
            "roc_auc": ex_auc,
        }
    )


exercise_df = pd.DataFrame(
    exercise_results
)

print()

print(
    exercise_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# PER-PERSON VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION BY PERSON")
print("=" * 70)


person_results = []

for person_id in sorted(
    validation_df[
        "person_id"
    ].unique()
):

    person_mask = (
        validation_df[
            "person_id"
        ]
        == person_id
    )

    person_true = (
        validation_df.loc[
            person_mask,
            "target"
        ]
    )

    person_pred = (
        validation_prediction[
            person_mask.to_numpy()
        ]
    )

    person_prob = (
        validation_probability[
            person_mask.to_numpy()
        ]
    )

    person_accuracy = accuracy_score(
        person_true,
        person_pred
    )

    person_f1 = f1_score(
        person_true,
        person_pred,
        zero_division=0
    )

    if person_true.nunique() == 2:
        person_auc = roc_auc_score(
            person_true,
            person_prob
        )
    else:
        person_auc = np.nan

    person_results.append(
        {
            "person_id": person_id,
            "samples": int(
                person_mask.sum()
            ),
            "accuracy": person_accuracy,
            "f1": person_f1,
            "roc_auc": person_auc,
        }
    )


person_df = pd.DataFrame(
    person_results
)

print()

print(
    person_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


model_package = {
    "model": model,
    "feature_columns": feature_columns,
    "label_map": LABEL_MAP,
    "threshold": THRESHOLD,
    "positive_class": "Incomplete",
    "negative_class": "Complete",
}


with open(
    MODEL_PATH,
    "wb"
) as file:

    pickle.dump(
        model_package,
        file
    )


# ============================================================
# SAVE VALIDATION PREDICTIONS
# ============================================================

validation_results.to_csv(
    VALIDATION_OUTPUT_PATH,
    index=False
)


# ============================================================
# FINISH
# ============================================================

print("\n" + "=" * 70)
print("SAVED")
print("=" * 70)

print("\nModel:")
print(MODEL_PATH)

print("\nValidation predictions:")
print(
    VALIDATION_OUTPUT_PATH
)

print("\nFeature count:")
print(
    len(feature_columns)
)

print("\nThreshold:")
print(
    THRESHOLD
)

print("\nPositive class:")
print(
    "Incomplete"
)

print("\n" + "=" * 70)
print("MOVEMENT STEP 7 COMPLETE")
print("=" * 70)