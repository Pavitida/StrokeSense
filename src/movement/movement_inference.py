"""Experimental movement inference for StrokeSense."""

from pathlib import Path
import argparse

import joblib
import mediapipe as mp
import numpy as np
import pandas as pd

from src.movement.movement_features import process_video


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "movement_model_final.pkl"

POSE_MODEL_PATH = (
    PROJECT_ROOT / "models" / "mediapipe" / "pose_landmarker.task"
)


def predict_movement(video_path):
    video_path = Path(video_path)

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    package = joblib.load(MODEL_PATH)

    model = package["model"]
    columns = package["feature_columns"]

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(POSE_MODEL_PATH)
        ),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Create a fresh landmarker for each video.
    with PoseLandmarker.create_from_options(options) as landmarker:
        features = process_video(str(video_path), landmarker)

    if features is None:
        raise ValueError(
            "Could not extract pose features from this video."
        )

    missing = [name for name in columns if name not in features]

    if missing:
        raise ValueError(f"Missing features: {missing}")

    values = pd.DataFrame(
        [{name: features[name] for name in columns}],
        columns=columns,
        dtype=np.float64,
    )

    # Keep preprocessing consistent with the trained model.
    # If the model is a Pipeline, its preprocessing runs internally.
    positive_class_name = package["positive_class"]
    positive_class = package["label_map"][positive_class_name]

    classes = list(model.classes_)

    if positive_class not in classes:
        raise ValueError(
            f"Positive class {positive_class!r} not in {classes}"
        )

    positive_index = classes.index(positive_class)

    score = float(
        model.predict_proba(values)[0][positive_index]
    )

    return {
        "status": "success",
        "score": score,
        "feature_count": len(columns),
        "positive_class": positive_class_name,
        "note": (
            "Experimental rehabilitation movement completeness "
            "classification; not a stroke diagnosis or stroke probability."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    args = parser.parse_args()

    print(predict_movement(args.video))
