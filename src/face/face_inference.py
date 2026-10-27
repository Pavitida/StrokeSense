"""Face inference for StrokeSense experimental prototype."""

from pathlib import Path
import argparse

import cv2
import joblib
import mediapipe as mp
import numpy as np

from src.face.face_features import extract_aligned_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FACE_MODEL_PATH = (
    PROJECT_ROOT / "models" / "face_landmark_model_v2.pkl"
)

LANDMARKER_PATH = (
    PROJECT_ROOT / "models" / "mediapipe" / "face_landmarker.task"
)


def predict_face(image_path):
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError("Cannot read image")

    package = joblib.load(FACE_MODEL_PATH)
    model = package["model"]
    feature_columns = package["feature_columns"]
    label_map = package["label_map"]

    positive_class = label_map["affected"]

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(LANDMARKER_PATH),
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.ascontiguousarray(rgb),
    )

    with mp.tasks.vision.FaceLandmarker.create_from_options(
        options
    ) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return {
            "status": "no_face_detected",
            "score": None,
        }

    features = extract_aligned_features(
        result.face_landmarks[0]
    )

    if features is None:
        return {
            "status": "invalid_landmarks",
            "score": None,
        }

    missing = [
        name for name in feature_columns
        if name not in features
    ]

    if missing:
        raise ValueError(f"Missing features: {missing}")

    values = np.array(
        [[features[name] for name in feature_columns]],
        dtype=np.float64,
    )

    if not np.isfinite(values).all():
        raise ValueError("Features contain NaN or infinity")

    class_indices = list(model.classes_)

    if positive_class not in class_indices:
        raise ValueError("Positive class not found in model")

    probability_index = class_indices.index(positive_class)

    score = float(
        model.predict_proba(values)[0][probability_index]
    )

    return {
        "status": "success",
        "score": score,
        "feature_count": len(feature_columns),
        "positive_class": "affected",
        "note": (
            "Experimental facial abnormality score; "
            "not a stroke diagnosis or stroke probability."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        required=True,
        help="Path to a face image",
    )
    args = parser.parse_args()

    print(predict_face(args.image))
