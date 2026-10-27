from pathlib import Path

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

DIAGNOSTIC_OUTPUT = (
    OUTPUT_DIR
    / "movement_diagnostic_results.csv"
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

QUALITY_THRESHOLDS = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("MOVEMENT STEP 7A - MODEL DIAGNOSTIC")
print("=" * 70)

df = pd.read_csv(FEATURE_PATH)

df["target"] = df["label"].map(LABEL_MAP)

print("\nSamples:")
print(len(df))

print("\nSplit:")
print(df["split"].value_counts())

print("\nLabels:")
print(df["label"].value_counts())

print("\nExercises:")
print(df["exercise"].value_counts())


# ============================================================
# IMPORTANT:
# TEST SET IS NOT USED IN THIS SCRIPT
# ============================================================

development_df = df[
    df["split"].isin(
        ["train", "validation"]
    )
].copy()

print("\nDevelopment samples:")
print(len(development_df))

print(
    "\nTest samples excluded:",
    (df["split"] == "test").sum()
)


# ============================================================
# FEATURE SELECTION
# ============================================================

EXCLUDE_COLUMNS = {
    "video_path",
    "filename",
    "person_id",
    "exercise",
    "label",
    "split",
    "original_split",
    "target",

    # video / detection quality
    "video_fps",
    "video_total_frames",
    "processed_frames",
    "detected_frames",
    "pose_detection_rate",
}


numeric_columns = (
    df.select_dtypes(include=[np.number])
    .columns
    .tolist()
)

feature_columns = [
    column
    for column in numeric_columns
    if column not in EXCLUDE_COLUMNS
]


print("\nMovement feature count:")
print(len(feature_columns))


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_model():

    return Pipeline(
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


def calculate_metrics(
    y_true,
    probabilities,
    threshold=0.5
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    result = {
        "accuracy":
            accuracy_score(
                y_true,
                predictions
            ),

        "precision":
            precision_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "f1":
            f1_score(
                y_true,
                predictions,
                zero_division=0
            ),
    }

    if len(np.unique(y_true)) == 2:

        result["roc_auc"] = (
            roc_auc_score(
                y_true,
                probabilities
            )
        )

    else:

        result["roc_auc"] = np.nan

    return result


diagnostic_results = []


# ============================================================
# DIAGNOSTIC 1
# PER-EXERCISE MODELS
# ============================================================

print("\n" + "=" * 70)
print("DIAGNOSTIC 1 - PER-EXERCISE MODELS")
print("=" * 70)


exercises = sorted(
    development_df[
        "exercise"
    ].unique()
)


for exercise in exercises:

    print("\n" + "-" * 70)
    print(exercise)
    print("-" * 70)

    exercise_df = development_df[
        development_df["exercise"]
        == exercise
    ].copy()

    train_df = exercise_df[
        exercise_df["split"]
        == "train"
    ].copy()

    validation_df = exercise_df[
        exercise_df["split"]
        == "validation"
    ].copy()

    print(
        "Train samples:",
        len(train_df)
    )

    print(
        "Validation samples:",
        len(validation_df)
    )

    print(
        "Train labels:"
    )
    print(
        train_df["label"]
        .value_counts()
    )

    print(
        "Validation labels:"
    )
    print(
        validation_df["label"]
        .value_counts()
    )

    if (
        len(train_df) == 0
        or len(validation_df) == 0
    ):
        print(
            "SKIPPED - missing split."
        )
        continue

    if train_df["target"].nunique() < 2:
        print(
            "SKIPPED - training set "
            "does not contain both labels."
        )
        continue

    X_train = (
        train_df[
            feature_columns
        ]
    )

    y_train = (
        train_df["target"]
    )

    X_val = (
        validation_df[
            feature_columns
        ]
    )

    y_val = (
        validation_df["target"]
    )

    model = create_model()

    model.fit(
        X_train,
        y_train
    )

    probabilities = (
        model.predict_proba(
            X_val
        )[:, 1]
    )

    metrics = calculate_metrics(
        y_val,
        probabilities,
        THRESHOLD
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

    diagnostic_results.append(
        {
            "diagnostic":
                "per_exercise",

            "exercise":
                exercise,

            "quality_threshold":
                np.nan,

            "train_samples":
                len(train_df),

            "validation_samples":
                len(validation_df),

            **metrics,
        }
    )


# ============================================================
# DIAGNOSTIC 2
# POSE QUALITY FILTERING
# ============================================================

print("\n" + "=" * 70)
print("DIAGNOSTIC 2 - POSE QUALITY FILTERING")
print("=" * 70)


for quality_threshold in QUALITY_THRESHOLDS:

    filtered_df = development_df[
        development_df[
            "pose_detection_rate"
        ] >= quality_threshold
    ].copy()

    train_df = filtered_df[
        filtered_df["split"]
        == "train"
    ].copy()

    validation_df = filtered_df[
        filtered_df["split"]
        == "validation"
    ].copy()

    print("\n" + "-" * 70)

    print(
        "Minimum pose detection rate:",
        quality_threshold
    )

    print(
        "Train:",
        len(train_df)
    )

    print(
        "Validation:",
        len(validation_df)
    )

    if (
        len(train_df) == 0
        or len(validation_df) == 0
    ):
        print("SKIPPED")
        continue

    if (
        train_df["target"].nunique() < 2
        or validation_df["target"].nunique() < 2
    ):
        print(
            "SKIPPED - both labels "
            "are required."
        )
        continue

    model = create_model()

    model.fit(
        train_df[feature_columns],
        train_df["target"]
    )

    probabilities = (
        model.predict_proba(
            validation_df[
                feature_columns
            ]
        )[:, 1]
    )

    metrics = calculate_metrics(
        validation_df["target"],
        probabilities,
        THRESHOLD
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

    diagnostic_results.append(
        {
            "diagnostic":
                "quality_filter",

            "exercise":
                "ALL",

            "quality_threshold":
                quality_threshold,

            "train_samples":
                len(train_df),

            "validation_samples":
                len(validation_df),

            **metrics,
        }
    )


# ============================================================
# DIAGNOSTIC 3
# PER-EXERCISE + QUALITY FILTER
# ============================================================

print("\n" + "=" * 70)
print(
    "DIAGNOSTIC 3 - "
    "PER-EXERCISE + QUALITY FILTER"
)
print("=" * 70)


for exercise in exercises:

    print("\n" + "#" * 70)
    print(exercise)
    print("#" * 70)

    for quality_threshold in [
        0.2,
        0.3,
        0.4,
    ]:

        subset = development_df[
            (
                development_df["exercise"]
                == exercise
            )
            &
            (
                development_df[
                    "pose_detection_rate"
                ]
                >= quality_threshold
            )
        ].copy()

        train_df = subset[
            subset["split"]
            == "train"
        ].copy()

        validation_df = subset[
            subset["split"]
            == "validation"
        ].copy()

        print(
            f"\nPose >= "
            f"{quality_threshold:.1f}"
        )

        print(
            "Train:",
            len(train_df),
            "| Validation:",
            len(validation_df)
        )

        if (
            len(train_df) < 10
            or len(validation_df) < 10
        ):
            print(
                "SKIPPED - too few samples."
            )
            continue

        if (
            train_df["target"].nunique()
            < 2
            or
            validation_df[
                "target"
            ].nunique()
            < 2
        ):
            print(
                "SKIPPED - both labels "
                "required."
            )
            continue

        model = create_model()

        model.fit(
            train_df[
                feature_columns
            ],
            train_df["target"]
        )

        probabilities = (
            model.predict_proba(
                validation_df[
                    feature_columns
                ]
            )[:, 1]
        )

        metrics = calculate_metrics(
            validation_df["target"],
            probabilities,
            THRESHOLD
        )

        print(
            f"Accuracy={metrics['accuracy']:.4f}"
            f" | F1={metrics['f1']:.4f}"
            f" | AUC={metrics['roc_auc']:.4f}"
        )

        diagnostic_results.append(
            {
                "diagnostic":
                    "exercise_quality",

                "exercise":
                    exercise,

                "quality_threshold":
                    quality_threshold,

                "train_samples":
                    len(train_df),

                "validation_samples":
                    len(validation_df),

                **metrics,
            }
        )


# ============================================================
# DIAGNOSTIC 4
# LOGISTIC REGRESSION COEFFICIENTS
# ============================================================

print("\n" + "=" * 70)
print("DIAGNOSTIC 4 - FEATURE COEFFICIENTS")
print("=" * 70)


train_df = development_df[
    development_df["split"]
    == "train"
].copy()

model = create_model()

model.fit(
    train_df[feature_columns],
    train_df["target"]
)


classifier = (
    model.named_steps[
        "classifier"
    ]
)

coefficients = (
    classifier.coef_[0]
)


coefficient_df = pd.DataFrame(
    {
        "feature":
            feature_columns,

        "coefficient":
            coefficients,

        "absolute_coefficient":
            np.abs(
                coefficients
            ),
    }
)

coefficient_df = (
    coefficient_df
    .sort_values(
        "absolute_coefficient",
        ascending=False
    )
    .reset_index(drop=True)
)


print(
    "\nTop 20 features "
    "by absolute coefficient:\n"
)

print(
    coefficient_df
    .head(20)
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

diagnostic_df = pd.DataFrame(
    diagnostic_results
)

diagnostic_df.to_csv(
    DIAGNOSTIC_OUTPUT,
    index=False
)


coefficient_path = (
    OUTPUT_DIR
    / "movement_feature_coefficients.csv"
)

coefficient_df.to_csv(
    coefficient_path,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)


if len(diagnostic_df) > 0:

    display_columns = [
        "diagnostic",
        "exercise",
        "quality_threshold",
        "train_samples",
        "validation_samples",
        "accuracy",
        "f1",
        "roc_auc",
    ]

    summary = (
        diagnostic_df[
            display_columns
        ]
        .sort_values(
            "roc_auc",
            ascending=False
        )
    )

    print()

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


print("\n" + "=" * 70)
print("SAVED")
print("=" * 70)

print(
    "\nDiagnostic results:"
)
print(
    DIAGNOSTIC_OUTPUT
)

print(
    "\nFeature coefficients:"
)
print(
    coefficient_path
)

print("\n" + "=" * 70)
print("MOVEMENT STEP 7A COMPLETE")
print("=" * 70)