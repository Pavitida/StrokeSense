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
    / "speech"
    / "torgo_mfcc_features.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = (
    MODEL_DIR
    / "speech_model.pkl"
)

PREDICTION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "speech"
    / "speech_validation_predictions.csv"
)


print("=" * 70)
print("SPEECH STEP 6 - TRAIN SPEECH MODEL")
print("=" * 70)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(FEATURE_PATH)

feature_columns = [
    c
    for c in df.columns
    if c.startswith("mfcc_")
]

print("\nTotal samples:")
print(len(df))

print("\nMFCC features:")
print(len(feature_columns))


# ============================================================
# TRAIN / VALIDATION
# ============================================================

train = df[
    df["split"] == "train"
].copy()

val = df[
    df["split"] == "validation"
].copy()


print("\n" + "=" * 70)
print("DATA")
print("=" * 70)

print("\nTrain:")
print("Samples:", len(train))
print("Speakers:", train["speaker_id"].nunique())

print("\nValidation:")
print("Samples:", len(val))
print("Speakers:", val["speaker_id"].nunique())


# ============================================================
# LABELS
# ============================================================

label_map = {
    "healthy": 0,
    "dysarthria": 1,
}

X_train = train[feature_columns].values
y_train = train["speech_status"].map(label_map).values

X_val = val[feature_columns].values
y_val = val["speech_status"].map(label_map).values


# ============================================================
# MODEL
# ============================================================

model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )
    )
])


print("\n" + "=" * 70)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 70)

model.fit(
    X_train,
    y_train
)


# ============================================================
# FRAME / UTTERANCE LEVEL VALIDATION
# ============================================================

val_probability = model.predict_proba(
    X_val
)[:, 1]

val_prediction = (
    val_probability >= 0.5
).astype(int)


print("\n" + "=" * 70)
print("UTTERANCE-LEVEL VALIDATION")
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
            "Healthy",
            "Dysarthria"
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
# SAVE VALIDATION PREDICTIONS
# ============================================================

val_output = val[
    [
        "speaker_id",
        "speech_status",
        "gender",
        "audio_path",
        "duration",
    ]
].copy()

val_output[
    "dysarthria_probability"
] = val_probability

val_output[
    "prediction"
] = val_prediction


# ============================================================
# SPEAKER LEVEL
# ============================================================

speaker_result = (
    val_output
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
    ] >= 0.5
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
print("SPEAKER-LEVEL VALIDATION")
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
# SAVE MODEL PACKAGE
# ============================================================

model_package = {
    "model": model,
    "feature_columns": feature_columns,
    "label_map": label_map,
    "threshold": 0.5,
    "target_sample_rate": 16000,
    "n_mfcc": 13,
}

with open(
    MODEL_PATH,
    "wb"
) as f:

    pickle.dump(
        model_package,
        f
    )


val_output.to_csv(
    PREDICTION_PATH,
    index=False
)


print("\n" + "=" * 70)
print("SAVED")
print("=" * 70)

print("\nModel:")
print(MODEL_PATH)

print("\nValidation predictions:")
print(PREDICTION_PATH)

print("\nSPEECH STEP 6 COMPLETE")