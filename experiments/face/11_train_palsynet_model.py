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
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "face_landmark"
    / "palsynet_split.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "face_landmark_model.pkl"
)

PREDICTION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "face_landmark"
    / "face_validation_predictions.csv"
)

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "left_eye_open",
    "right_eye_open",
    "eye_open_diff",

    "left_brow_height",
    "right_brow_height",
    "eyebrow_height_diff",

    "mouth_width",
    "mouth_open",
    "mouth_height_diff",
    "mouth_slope",

    "mouth_left_nose",
    "mouth_right_nose",
    "mouth_nose_diff",

    "left_eye_nose",
    "right_eye_nose",
    "eye_nose_diff",

    "face_length",
    "overall_asymmetry"
]


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("STEP 11 - TRAIN MEDIAPIPE FACE MODEL")
print("=" * 70)

df = pd.read_csv(INPUT_PATH)

train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()


print("\nTrain:")
print("Frames:", len(train_df))
print("People:", train_df["person_id"].nunique())

print("\nValidation:")
print("Frames:", len(val_df))
print("People:", val_df["person_id"].nunique())


# ============================================================
# LABEL
#
# affected   = 1
# unaffected = 0
# ============================================================

label_map = {
    "unaffected": 0,
    "affected": 1
}

y_train = (
    train_df["label"]
    .map(label_map)
    .values
)

y_val = (
    val_df["label"]
    .map(label_map)
    .values
)

X_train = train_df[FEATURES].values
X_val = val_df[FEATURES].values


# ============================================================
# MODEL
# ============================================================

model = Pipeline(
    [
        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 70)

model.fit(
    X_train,
    y_train
)


# ============================================================
# FRAME-LEVEL VALIDATION
# ============================================================

val_probability = (
    model.predict_proba(X_val)[:, 1]
)

val_prediction = (
    val_probability >= 0.5
).astype(int)


print("\n" + "=" * 70)
print("FRAME-LEVEL VALIDATION")
print("=" * 70)


print(
    "\nAccuracy:",
    f"{accuracy_score(y_val, val_prediction):.4f}"
)

print(
    "Precision:",
    f"{precision_score(y_val, val_prediction):.4f}"
)

print(
    "Recall:",
    f"{recall_score(y_val, val_prediction):.4f}"
)

print(
    "F1-score:",
    f"{f1_score(y_val, val_prediction):.4f}"
)

print(
    "ROC-AUC:",
    f"{roc_auc_score(y_val, val_probability):.4f}"
)


print("\nClassification report:")

print(
    classification_report(
        y_val,
        val_prediction,
        target_names=[
            "Unaffected",
            "Affected"
        ],
        digits=4
    )
)


print("Confusion matrix:")

print(
    confusion_matrix(
        y_val,
        val_prediction
    )
)


# ============================================================
# PERSON-LEVEL VALIDATION
#
# Average frame probabilities for each person.
# ============================================================

prediction_df = val_df[
    [
        "person_id",
        "label",
        "frame_number"
    ]
].copy()

prediction_df["true_label"] = y_val

prediction_df[
    "affected_probability"
] = val_probability


person_df = (
    prediction_df
    .groupby(
        [
            "person_id",
            "label",
            "true_label"
        ],
        as_index=False
    )
    .agg(
        affected_probability=(
            "affected_probability",
            "mean"
        ),

        frames=(
            "frame_number",
            "count"
        )
    )
)


person_df["prediction"] = (
    person_df[
        "affected_probability"
    ] >= 0.5
).astype(int)


person_true = (
    person_df["true_label"].values
)

person_pred = (
    person_df["prediction"].values
)

person_prob = (
    person_df[
        "affected_probability"
    ].values
)


print("\n" + "=" * 70)
print("PERSON-LEVEL VALIDATION")
print("=" * 70)


print("\nPeople:")
print(len(person_df))


print(
    "\nAccuracy:",
    f"{accuracy_score(person_true, person_pred):.4f}"
)

print(
    "Precision:",
    f"{precision_score(person_true, person_pred):.4f}"
)

print(
    "Recall:",
    f"{recall_score(person_true, person_pred):.4f}"
)

print(
    "F1-score:",
    f"{f1_score(person_true, person_pred):.4f}"
)

print(
    "ROC-AUC:",
    f"{roc_auc_score(person_true, person_prob):.4f}"
)


print("\nPerson predictions:")

print(
    person_df[
        [
            "person_id",
            "label",
            "frames",
            "affected_probability",
            "prediction"
        ]
    ].to_string(index=False)
)


print("\nConfusion matrix:")

print(
    confusion_matrix(
        person_true,
        person_pred
    )
)


# ============================================================
# SAVE VALIDATION PREDICTIONS
# ============================================================

person_df.to_csv(
    PREDICTION_PATH,
    index=False
)


# ============================================================
# SAVE MODEL
# ============================================================

model_package = {

    "model": model,

    "features": FEATURES,

    "label_map": label_map,

    "threshold": 0.5,

    "positive_class": "affected",

    "description":
        "MediaPipe facial asymmetry model "
        "for facial palsy/weakness screening"
}


joblib.dump(
    model_package,
    MODEL_PATH
)


print("\n" + "=" * 70)
print("SAVED")
print("=" * 70)

print("\nModel:")
print(MODEL_PATH)

print("\nValidation predictions:")
print(PREDICTION_PATH)

print("\nSTEP 11 COMPLETE")