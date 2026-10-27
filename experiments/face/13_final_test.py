from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "face_landmark"
    / "palsynet_aligned_features.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "face_landmark_model_v2.pkl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "face_landmark"
    / "face_final_test_predictions.csv"
)

# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("STEP 13 - FINAL FACE MODEL TEST")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

package = joblib.load(MODEL_PATH)

model = package["model"]
features = package["feature_columns"]
threshold = package["threshold"]

print("\nModel:")
print(MODEL_PATH)

print("\nFeature count:")
print(len(features))

print("\nThreshold:")
print(threshold)

# ============================================================
# TEST SET ONLY
# ============================================================

test_df = df[
    df["split"] == "test"
].copy()

print("\n" + "=" * 70)
print("TEST DATA")
print("=" * 70)

print("\nFrames:")
print(len(test_df))

print("\nPeople:")
print(test_df["person_id"].nunique())

print("\nPeople by label:")

print(
    test_df[
        ["person_id", "label"]
    ]
    .drop_duplicates()
    ["label"]
    .value_counts()
)

# ============================================================
# LABELS
# ============================================================

label_map = {
    "unaffected": 0,
    "affected": 1
}

y_test = (
    test_df["label"]
    .map(label_map)
    .values
)

X_test = test_df[features].values

# ============================================================
# PREDICT
# ============================================================

probability = (
    model.predict_proba(X_test)[:, 1]
)

prediction = (
    probability >= threshold
).astype(int)

# ============================================================
# FRAME LEVEL
# ============================================================

print("\n" + "=" * 70)
print("FRAME-LEVEL FINAL TEST")
print("=" * 70)

print(
    "\nAccuracy:",
    f"{accuracy_score(y_test, prediction):.4f}"
)

print(
    "Precision:",
    f"{precision_score(y_test, prediction, zero_division=0):.4f}"
)

print(
    "Recall:",
    f"{recall_score(y_test, prediction, zero_division=0):.4f}"
)

print(
    "F1-score:",
    f"{f1_score(y_test, prediction, zero_division=0):.4f}"
)

print(
    "ROC-AUC:",
    f"{roc_auc_score(y_test, probability):.4f}"
)

print("\nClassification report:\n")

print(
    classification_report(
        y_test,
        prediction,
        target_names=[
            "Unaffected",
            "Affected"
        ],
        digits=4,
        zero_division=0
    )
)

print("Confusion matrix:")

print(
    confusion_matrix(
        y_test,
        prediction
    )
)

# ============================================================
# PERSON LEVEL
# ============================================================

results = test_df[
    [
        "person_id",
        "label"
    ]
].copy()

results["true_label"] = y_test

results[
    "affected_probability"
] = probability

person_results = (
    results
    .groupby(
        [
            "person_id",
            "label",
            "true_label"
        ],
        as_index=False
    )
    .agg(
        frames=(
            "affected_probability",
            "size"
        ),
        affected_probability=(
            "affected_probability",
            "mean"
        )
    )
)

person_results["prediction"] = (
    person_results[
        "affected_probability"
    ] >= threshold
).astype(int)

person_true = (
    person_results[
        "true_label"
    ].values
)

person_pred = (
    person_results[
        "prediction"
    ].values
)

person_prob = (
    person_results[
        "affected_probability"
    ].values
)

# ============================================================
# PERSON METRICS
# ============================================================

print("\n" + "=" * 70)
print("PERSON-LEVEL FINAL TEST")
print("=" * 70)

print("\nPeople:")
print(len(person_results))

print(
    "\nAccuracy:",
    f"{accuracy_score(person_true, person_pred):.4f}"
)

print(
    "Precision:",
    f"{precision_score(person_true, person_pred, zero_division=0):.4f}"
)

print(
    "Recall:",
    f"{recall_score(person_true, person_pred, zero_division=0):.4f}"
)

print(
    "F1-score:",
    f"{f1_score(person_true, person_pred, zero_division=0):.4f}"
)

print(
    "ROC-AUC:",
    f"{roc_auc_score(person_true, person_prob):.4f}"
)

print("\nPerson predictions:\n")

print(
    person_results[
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
# SAVE
# ============================================================

person_results.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 70)
print("SAVED")
print("=" * 70)

print(OUTPUT_PATH)

print("\nSTEP 13 COMPLETE")