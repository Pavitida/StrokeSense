from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "movement"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "movement_model_comparison.csv"
)

# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

LABEL_MAP = {
    "Complete": 0,
    "Incomplete": 1,
}

QUALITY_THRESHOLDS = [
    0.3,
    0.4,
    0.5,
]

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
print("MOVEMENT STEP 7B - MODEL COMPARISON")
print("=" * 70)

df = pd.read_csv(FEATURE_PATH)

df["target"] = df["label"].map(LABEL_MAP)

# IMPORTANT:
# Test set is completely excluded during model selection.
dev_df = df[
    df["split"].isin(["train", "validation"])
].copy()

test_count = (
    df["split"] == "test"
).sum()

print("\nDevelopment samples:", len(dev_df))
print("Test samples excluded:", test_count)

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

print("Feature count:", len(feature_columns))

# ============================================================
# MODELS
# ============================================================

def make_logistic():

    return Pipeline(
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


def make_random_forest():

    return RandomForestClassifier(
        n_estimators=500,
        max_depth=5,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


MODELS = {
    "LogisticRegression":
        make_logistic,

    "RandomForest":
        make_random_forest,
}

# ============================================================
# METRICS
# ============================================================

def evaluate(
    y_true,
    probability,
    threshold=0.5
):

    prediction = (
        probability >= threshold
    ).astype(int)

    return {
        "accuracy":
            accuracy_score(
                y_true,
                prediction
            ),

        "precision":
            precision_score(
                y_true,
                prediction,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_true,
                prediction,
                zero_division=0
            ),

        "f1":
            f1_score(
                y_true,
                prediction,
                zero_division=0
            ),

        "roc_auc":
            roc_auc_score(
                y_true,
                probability
            ),
    }

# ============================================================
# RUN COMPARISON
# ============================================================

results = []

for quality_threshold in QUALITY_THRESHOLDS:

    print("\n" + "=" * 70)
    print(
        "POSE QUALITY >=",
        quality_threshold
    )
    print("=" * 70)

    subset = dev_df[
        dev_df["pose_detection_rate"]
        >= quality_threshold
    ].copy()

    train_df = subset[
        subset["split"] == "train"
    ].copy()

    val_df = subset[
        subset["split"] == "validation"
    ].copy()

    print(
        "Train:",
        len(train_df)
    )

    print(
        "Validation:",
        len(val_df)
    )

    print("\nTrain labels:")
    print(
        train_df["label"].value_counts()
    )

    print("\nValidation labels:")
    print(
        val_df["label"].value_counts()
    )

    X_train = train_df[
        feature_columns
    ]

    y_train = train_df[
        "target"
    ]

    X_val = val_df[
        feature_columns
    ]

    y_val = val_df[
        "target"
    ]

    for model_name, factory in MODELS.items():

        print(
            "\n---",
            model_name,
            "---"
        )

        model = factory()

        model.fit(
            X_train,
            y_train
        )

        probability = (
            model.predict_proba(
                X_val
            )[:, 1]
        )

        metrics = evaluate(
            y_val,
            probability
        )

        print(
            f"Accuracy : "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Precision: "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"Recall   : "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"F1       : "
            f"{metrics['f1']:.4f}"
        )

        print(
            f"ROC-AUC  : "
            f"{metrics['roc_auc']:.4f}"
        )

        results.append(
            {
                "model":
                    model_name,

                "pose_threshold":
                    quality_threshold,

                "train_samples":
                    len(train_df),

                "validation_samples":
                    len(val_df),

                **metrics,
            }
        )

# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = (
    results_df
    .sort_values(
        [
            "roc_auc",
            "f1",
        ],
        ascending=False
    )
    .reset_index(drop=True)
)

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 70)
print("SAVED")
print("=" * 70)

print(OUTPUT_PATH)

print("\nIMPORTANT:")
print(
    "The test set has NOT been used "
    "for model selection."
)

print("=" * 70)
print("MOVEMENT STEP 7B COMPLETE")
print("=" * 70)