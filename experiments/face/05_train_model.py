from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_PATH = PROJECT_ROOT / "outputs" / "face"
MODEL_PATH = PROJECT_ROOT / "models"

MODEL_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# LOAD DATA
# =========================================================

print("=" * 70)
print("STEP 5 - TRAIN FACE MODEL")
print("=" * 70)

X_train = np.load(
    OUTPUT_PATH / "X_train.npy"
)

X_val = np.load(
    OUTPUT_PATH / "X_val.npy"
)

train_df = pd.read_csv(
    OUTPUT_PATH / "train_metadata.csv"
)

val_df = pd.read_csv(
    OUTPUT_PATH / "val_metadata.csv"
)


y_train = train_df["severity"].astype(str)
y_val = val_df["severity"].astype(str)


print("\nTrain features:", X_train.shape)
print("Validation features:", X_val.shape)

print("\nTrain labels:")
print(y_train.value_counts())

print("\nValidation labels:")
print(y_val.value_counts())


# =========================================================
# MODEL
# =========================================================

print("\n" + "=" * 70)
print("TRAINING LINEAR SVM")
print("=" * 70)

model = LinearSVC(
    C=1.0,
    class_weight="balanced",
    max_iter=10000,
    random_state=42
)

start_time = time.time()

model.fit(
    X_train,
    y_train
)

training_time = (
    time.time() - start_time
)

print(
    f"\nTraining time: "
    f"{training_time:.2f} seconds"
)


# =========================================================
# VALIDATION PREDICTION
# =========================================================

print("\nPredicting validation set...")

y_pred = model.predict(
    X_val
)


# =========================================================
# METRICS
# =========================================================

accuracy = accuracy_score(
    y_val,
    y_pred
)

macro_precision = precision_score(
    y_val,
    y_pred,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    y_val,
    y_pred,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_val,
    y_pred,
    average="macro",
    zero_division=0
)


print("\n" + "=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)

print(
    f"Accuracy        : {accuracy:.4f}"
)

print(
    f"Macro Precision : {macro_precision:.4f}"
)

print(
    f"Macro Recall    : {macro_recall:.4f}"
)

print(
    f"Macro F1-score  : {macro_f1:.4f}"
)


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_val,
    y_pred,
    digits=4,
    zero_division=0
)

print(report)


# =========================================================
# CONFUSION MATRIX
# =========================================================

classes = sorted(
    y_train.unique()
)

cm = confusion_matrix(
    y_val,
    y_pred,
    labels=classes
)


print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm_df = pd.DataFrame(
    cm,
    index=classes,
    columns=classes
)

print(cm_df)


# =========================================================
# SAVE VALIDATION PREDICTIONS
# =========================================================

prediction_df = val_df.copy()

prediction_df[
    "predicted_severity"
] = y_pred

prediction_df[
    "correct"
] = (
    prediction_df["severity"]
    == prediction_df["predicted_severity"]
)

prediction_df.to_csv(
    OUTPUT_PATH
    / "validation_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# SAVE METRICS
# =========================================================

metrics_df = pd.DataFrame(
    [{
        "model": "LinearSVC",
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "training_time_seconds": training_time
    }]
)

metrics_df.to_csv(
    OUTPUT_PATH
    / "validation_metrics.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# SAVE MODEL
# =========================================================

MODEL_FILE = (
    MODEL_PATH
    / "face_model.pkl"
)

joblib.dump(
    model,
    MODEL_FILE
)


# =========================================================
# FINAL
# =========================================================

print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(MODEL_FILE)

print("\nSTEP 5 COMPLETE")