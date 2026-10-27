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

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "movement_model_final.pkl"
)

PREDICTION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "movement"
    / "movement_test_predictions.csv"
)

EXERCISE_RESULT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "movement"
    / "movement_test_by_exercise.csv"
)

# ============================================================
# SETTINGS - LOCKED BEFORE TEST
# ============================================================

POSE_THRESHOLD = 0.4
CLASSIFICATION_THRESHOLD = 0.5
RANDOM_STATE = 42

LABEL_MAP = {
    "Complete": 0,
    "Incomplete": 1,
}

EXCLUDE_COLUMNS = {
    "video_path",
    "filename",
    "person_id",
    "exercise",
    "label",
    "split",
    "original_split",
    "target",
    "video_fps",
    "video_total_frames",
    "processed_frames",
    "detected_frames",
    "pose_detection_rate",
}

# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("MOVEMENT STEP 8 - FINAL MODEL AND TEST")
print("=" * 70)

df = pd.read_csv(FEATURE_PATH)

df["target"] = df["label"].map(LABEL_MAP)

print("\nOriginal samples:", len(df))

print("\nOriginal split:")
print(df["split"].value_counts())

# ============================================================
# QUALITY FILTER
# ============================================================

df = df[
    df["pose_detection_rate"] >= POSE_THRESHOLD
].copy()

print("\nPose threshold:", POSE_THRESHOLD)
print("Samples after quality filter:", len(df))

print("\nFiltered split:")
print(df["split"].value_counts())

# ============================================================
# FINAL DEVELOPMENT / TEST
# ============================================================

development_df = df[
    df["split"].isin(
        ["train", "validation"]
    )
].copy()

test_df = df[
    df["split"] == "test"
].copy()

print("\n" + "=" * 70)
print("FINAL DATA")
print("=" * 70)

print("\nDevelopment:")
print("Samples:", len(development_df))
print("People:", development_df["person_id"].nunique())
print("Person IDs:", sorted(
    development_df["person_id"]
    .astype(str)
    .unique()
))

print("\nLabels:")
print(development_df["label"].value_counts())

print("\nTest:")
print("Samples:", len(test_df))
print("People:", test_df["person_id"].nunique())
print("Person IDs:", sorted(
    test_df["person_id"]
    .astype(str)
    .unique()
))

print("\nLabels:")
print(test_df["label"].value_counts())

# ============================================================
# LEAKAGE CHECK
# ============================================================

dev_people = set(
    development_df["person_id"]
    .astype(str)
)

test_people = set(
    test_df["person_id"]
    .astype(str)
)

overlap = dev_people.intersection(
    test_people
)

print("\n" + "=" * 70)
print("LEAKAGE CHECK")
print("=" * 70)

print(
    "Development / Test person overlap:",
    overlap
)

if overlap:
    raise RuntimeError(
        "Person leakage detected."
    )

print("PASS - no person leakage.")

# ============================================================
# FEATURES
# ============================================================

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

print("\nFeature count:", len(feature_columns))

X_dev = development_df[
    feature_columns
]

y_dev = development_df[
    "target"
]

X_test = test_df[
    feature_columns
]

y_test = test_df[
    "target"
]

# ============================================================
# FINAL MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAIN FINAL LOGISTIC REGRESSION")
print("=" * 70)

model = Pipeline(
    [
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
    ]
)

model.fit(
    X_dev,
    y_dev
)

print("Training complete.")

# ============================================================
# TEST - FIRST FINAL EVALUATION
# ============================================================

probability = (
    model.predict_proba(
        X_test
    )[:, 1]
)

prediction = (
    probability
    >= CLASSIFICATION_THRESHOLD
).astype(int)

accuracy = accuracy_score(
    y_test,
    prediction
)

precision = precision_score(
    y_test,
    prediction,
    zero_division=0
)

recall = recall_score(
    y_test,
    prediction,
    zero_division=0
)

f1 = f1_score(
    y_test,
    prediction,
    zero_division=0
)

auc = roc_auc_score(
    y_test,
    probability
)

print("\n" + "=" * 70)
print("FINAL TEST RESULT")
print("=" * 70)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")

print("\nClassification report:")

print(
    classification_report(
        y_test,
        prediction,
        target_names=[
            "Complete",
            "Incomplete"
        ],
        digits=4,
        zero_division=0,
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
# TEST BY EXERCISE
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST BY EXERCISE")
print("=" * 70)

exercise_results = []

for exercise in sorted(
    test_df["exercise"].unique()
):

    mask = (
        test_df["exercise"]
        == exercise
    )

    subset_true = (
        y_test.loc[mask]
    )

    subset_prob = (
        probability[
            mask.to_numpy()
        ]
    )

    subset_pred = (
        prediction[
            mask.to_numpy()
        ]
    )

    result = {
        "exercise": exercise,
        "samples": len(subset_true),

        "accuracy":
            accuracy_score(
                subset_true,
                subset_pred
            ),

        "precision":
            precision_score(
                subset_true,
                subset_pred,
                zero_division=0
            ),

        "recall":
            recall_score(
                subset_true,
                subset_pred,
                zero_division=0
            ),

        "f1":
            f1_score(
                subset_true,
                subset_pred,
                zero_division=0
            ),
    }

    if subset_true.nunique() == 2:
        result["roc_auc"] = (
            roc_auc_score(
                subset_true,
                subset_prob
            )
        )
    else:
        result["roc_auc"] = np.nan

    exercise_results.append(
        result
    )

exercise_results_df = pd.DataFrame(
    exercise_results
)

print(
    exercise_results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = (
    test_df[
        [
            "person_id",
            "exercise",
            "label",
            "split",
            "pose_detection_rate",
        ]
    ]
    .copy()
)

if "filename" in test_df.columns:
    prediction_df.insert(
        0,
        "filename",
        test_df["filename"].values
    )

prediction_df[
    "true_target"
] = y_test.values

prediction_df[
    "probability_incomplete"
] = probability

prediction_df[
    "predicted_target"
] = prediction

prediction_df[
    "predicted_label"
] = np.where(
    prediction == 1,
    "Incomplete",
    "Complete",
)

# ============================================================
# SAVE FINAL MODEL
# ============================================================

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

PREDICTION_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

model_package = {
    "model":
        model,

    "feature_columns":
        feature_columns,

    "pose_threshold":
        POSE_THRESHOLD,

    "classification_threshold":
        CLASSIFICATION_THRESHOLD,

    "positive_class":
        "Incomplete",

    "label_map":
        LABEL_MAP,
}

with open(
    MODEL_PATH,
    "wb"
) as file:
    pickle.dump(
        model_package,
        file
    )

prediction_df.to_csv(
    PREDICTION_PATH,
    index=False
)

exercise_results_df.to_csv(
    EXERCISE_RESULT_PATH,
    index=False
)

# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("SAVED")
print("=" * 70)

print("\nFinal model:")
print(MODEL_PATH)

print("\nTest predictions:")
print(PREDICTION_PATH)

print("\nTest by exercise:")
print(EXERCISE_RESULT_PATH)

print("\nConfiguration:")
print(
    "Pose threshold:",
    POSE_THRESHOLD
)
print(
    "Classification threshold:",
    CLASSIFICATION_THRESHOLD
)
print(
    "Positive class: Incomplete"
)

print("\n" + "=" * 70)
print("MOVEMENT STEP 8 COMPLETE")
print("=" * 70)