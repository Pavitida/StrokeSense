from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "face_landmark"
    / "palsynet_aligned_features.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "face_landmark"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "face_landmark_model_v2.pkl"

PREDICTION_PATH = (
    OUTPUT_DIR
    / "face_validation_predictions_v2.csv"
)

# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("STEP 12B - TRAIN ALIGNED FACE MODEL V2")
print("=" * 70)

df = pd.read_csv(FEATURE_PATH)

print("\nTotal frames:")
print(len(df))

print("\nPeople:")
print(df["person_id"].nunique())

print("\nSplit:")
print(df["split"].value_counts())


# ============================================================
# FEATURE COLUMNS
# ============================================================

NON_FEATURE_COLUMNS = {
    "file_path",
    "relative_path",
    "file_name",
    "image_path",
    "video_file",

    "label",
    "person_id",
    "split",

    "frame",
    "frame_id",
    "frame_number",
}

feature_cols = [
    col
    for col in df.columns
    if col not in NON_FEATURE_COLUMNS
    and pd.api.types.is_numeric_dtype(df[col])
]
assert "frame_number" not in feature_cols, \
    "ERROR: frame_number leaked into model features"

assert len(feature_cols) == 29, \
    f"Expected 29 facial features, got {len(feature_cols)}"

print("\nFeature count:")
print(len(feature_cols))

print("\nFeatures:")
for feature in feature_cols:
    print("•", feature)


# ============================================================
# LABEL
# ============================================================

label_map = {
    "unaffected": 0,
    "affected": 1,
}

df["target"] = df["label"].map(label_map)

if df["target"].isna().any():
    raise ValueError(
        "Unknown label found. Expected affected/unaffected."
    )


# ============================================================
# TRAIN / VALIDATION
# ============================================================

train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()

X_train = train_df[feature_cols].values
y_train = train_df["target"].values

X_val = val_df[feature_cols].values
y_val = val_df["target"].values

print("\n" + "=" * 70)
print("DATA")
print("=" * 70)

print("\nTrain:")
print("Frames:", len(train_df))
print("People:", train_df["person_id"].nunique())

print("\nValidation:")
print("Frames:", len(val_df))
print("People:", val_df["person_id"].nunique())


# ============================================================
# MODEL
# ============================================================

model = Pipeline(
    [
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]
)

print("\n" + "=" * 70)
print("TRAINING LOGISTIC REGRESSION V2")
print("=" * 70)

model.fit(X_train, y_train)


# ============================================================
# FRAME LEVEL
# ============================================================

val_probability = model.predict_proba(X_val)[:, 1]

val_prediction = (
    val_probability >= 0.5
).astype(int)

print("\n" + "=" * 70)
print("FRAME-LEVEL VALIDATION")
print("=" * 70)

frame_accuracy = accuracy_score(
    y_val,
    val_prediction,
)

frame_precision = precision_score(
    y_val,
    val_prediction,
    zero_division=0,
)

frame_recall = recall_score(
    y_val,
    val_prediction,
    zero_division=0,
)

frame_f1 = f1_score(
    y_val,
    val_prediction,
    zero_division=0,
)

frame_auc = roc_auc_score(
    y_val,
    val_probability,
)

print(f"\nAccuracy: {frame_accuracy:.4f}")
print(f"Precision: {frame_precision:.4f}")
print(f"Recall: {frame_recall:.4f}")
print(f"F1-score: {frame_f1:.4f}")
print(f"ROC-AUC: {frame_auc:.4f}")

print("\nClassification report:\n")

print(
    classification_report(
        y_val,
        val_prediction,
        target_names=[
            "Unaffected",
            "Affected",
        ],
        zero_division=0,
    )
)

print("Confusion matrix:")
print(
    confusion_matrix(
        y_val,
        val_prediction,
    )
)


# ============================================================
# PERSON LEVEL
# ============================================================

val_result = val_df[
    [
        "person_id",
        "label",
    ]
].copy()

val_result["target"] = y_val
val_result["affected_probability"] = val_probability
val_result["frame_prediction"] = val_prediction

person_result = (
    val_result
    .groupby(
        [
            "person_id",
            "label",
            "target",
        ],
        as_index=False,
    )
    .agg(
        frames=(
            "affected_probability",
            "size",
        ),
        affected_probability=(
            "affected_probability",
            "mean",
        ),
    )
)

person_result["prediction"] = (
    person_result["affected_probability"] >= 0.5
).astype(int)

person_y = person_result["target"].values
person_probability = person_result[
    "affected_probability"
].values
person_prediction = person_result[
    "prediction"
].values


print("\n" + "=" * 70)
print("PERSON-LEVEL VALIDATION")
print("=" * 70)

print("\nPeople:")
print(len(person_result))

person_accuracy = accuracy_score(
    person_y,
    person_prediction,
)

person_precision = precision_score(
    person_y,
    person_prediction,
    zero_division=0,
)

person_recall = recall_score(
    person_y,
    person_prediction,
    zero_division=0,
)

person_f1 = f1_score(
    person_y,
    person_prediction,
    zero_division=0,
)

person_auc = roc_auc_score(
    person_y,
    person_probability,
)

print(f"\nAccuracy: {person_accuracy:.4f}")
print(f"Precision: {person_precision:.4f}")
print(f"Recall: {person_recall:.4f}")
print(f"F1-score: {person_f1:.4f}")
print(f"ROC-AUC: {person_auc:.4f}")

print("\nPerson predictions:\n")

print(
    person_result[
        [
            "person_id",
            "label",
            "frames",
            "affected_probability",
            "prediction",
        ]
    ].to_string(index=False)
)

print("\nConfusion matrix:")

print(
    confusion_matrix(
        person_y,
        person_prediction,
    )
)


# ============================================================
# SAVE
# ============================================================

model_package = {
    "model": model,
    "feature_columns": feature_cols,
    "threshold": 0.5,
    "label_map": label_map,
    "version": "aligned_face_v2",
}

joblib.dump(
    model_package,
    MODEL_PATH,
)

person_result.to_csv(
    PREDICTION_PATH,
    index=False,
)


print("\n" + "=" * 70)
print("SAVED")
print("=" * 70)

print("\nModel:")
print(MODEL_PATH)

print("\nValidation predictions:")
print(PREDICTION_PATH)

print("\nSTEP 12B COMPLETE")