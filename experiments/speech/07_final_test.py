from pathlib import Path
import pickle

import numpy as np
import pandas as pd

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
    / "speech"
    / "torgo_mfcc_features.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "speech_model.pkl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "speech"
    / "speech_final_test_predictions.csv"
)


print("=" * 70)
print("SPEECH STEP 7 - FINAL SPEECH MODEL TEST")
print("=" * 70)


# ============================================================
# LOAD MODEL
# ============================================================

with open(MODEL_PATH, "rb") as f:
    package = pickle.load(f)

model = package["model"]
feature_columns = package["feature_columns"]
label_map = package["label_map"]
threshold = package["threshold"]


print("\nModel:")
print(MODEL_PATH)

print("\nFeature count:")
print(len(feature_columns))

print("\nThreshold:")
print(threshold)


# ============================================================
# LOAD TEST DATA ONLY
# ============================================================

df = pd.read_csv(FEATURE_PATH)

test = df[
    df["split"] == "test"
].copy()

print("\n" + "=" * 70)
print("TEST DATA")
print("=" * 70)

print("\nUtterances:")
print(len(test))

print("\nSpeakers:")
print(test["speaker_id"].nunique())

print("\nSpeakers by label:")

print(
    test[
        ["speaker_id", "speech_status"]
    ]
    .drop_duplicates()
    ["speech_status"]
    .value_counts()
)

print("\nUtterances by label:")
print(test["speech_status"].value_counts())


# ============================================================
# PREPARE TEST
# ============================================================

X_test = test[feature_columns].values

y_test = (
    test["speech_status"]
    .map(label_map)
    .values
)


# ============================================================
# PREDICT
# ============================================================

probability = model.predict_proba(
    X_test
)[:, 1]

prediction = (
    probability >= threshold
).astype(int)


# ============================================================
# UTTERANCE LEVEL
# ============================================================

print("\n" + "=" * 70)
print("UTTERANCE-LEVEL FINAL TEST")
print("=" * 70)

print(
    "\nAccuracy:",
    f"{accuracy_score(y_test, prediction):.4f}"
)

print(
    "Precision:",
    f"{precision_score(y_test, prediction):.4f}"
)

print(
    "Recall:",
    f"{recall_score(y_test, prediction):.4f}"
)

print(
    "F1-score:",
    f"{f1_score(y_test, prediction):.4f}"
)

print(
    "ROC-AUC:",
    f"{roc_auc_score(y_test, probability):.4f}"
)

print("\nClassification report:")

print(
    classification_report(
        y_test,
        prediction,
        target_names=[
            "Healthy",
            "Dysarthria"
        ],
        digits=4
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
# SAVE UTTERANCE PREDICTIONS
# ============================================================

output = test[
    [
        "speaker_id",
        "speech_status",
        "gender",
        "audio_path",
        "duration",
    ]
].copy()

output[
    "dysarthria_probability"
] = probability

output[
    "prediction"
] = prediction


# ============================================================
# SPEAKER LEVEL
# ============================================================

speaker_result = (
    output
    .groupby(
        [
            "speaker_id",
            "speech_status"
        ],
        as_index=False
    )
    .agg(
        utterances=(
            "audio_path",
            "size"
        ),
        dysarthria_probability=(
            "dysarthria_probability",
            "mean"
        )
    )
)

speaker_result[
    "true_label"
] = (
    speaker_result[
        "speech_status"
    ]
    .map(label_map)
)

speaker_result[
    "prediction"
] = (
    speaker_result[
        "dysarthria_probability"
    ] >= threshold
).astype(int)


speaker_y = speaker_result[
    "true_label"
].values

speaker_pred = speaker_result[
    "prediction"
].values

speaker_prob = speaker_result[
    "dysarthria_probability"
].values


print("\n" + "=" * 70)
print("SPEAKER-LEVEL FINAL TEST")
print("=" * 70)

print("\nSpeakers:")
print(len(speaker_result))

print(
    "\nAccuracy:",
    f"{accuracy_score(speaker_y, speaker_pred):.4f}"
)

print(
    "Precision:",
    f"{precision_score(speaker_y, speaker_pred, zero_division=0):.4f}"
)

print(
    "Recall:",
    f"{recall_score(speaker_y, speaker_pred, zero_division=0):.4f}"
)

print(
    "F1-score:",
    f"{f1_score(speaker_y, speaker_pred, zero_division=0):.4f}"
)

if len(np.unique(speaker_y)) == 2:
    print(
        "ROC-AUC:",
        f"{roc_auc_score(speaker_y, speaker_prob):.4f}"
    )


print("\nSpeaker predictions:")

print(
    speaker_result[
        [
            "speaker_id",
            "speech_status",
            "utterances",
            "dysarthria_probability",
            "prediction",
        ]
    ].to_string(index=False)
)

print("\nConfusion matrix:")

print(
    confusion_matrix(
        speaker_y,
        speaker_pred
    )
)


# ============================================================
# SAVE
# ============================================================

output.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 70)
print("SAVED")
print("=" * 70)

print(OUTPUT_PATH)

print("\nSPEECH STEP 7 COMPLETE")