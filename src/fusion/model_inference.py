from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

MODEL_FILES = {
    "face": "face_landmark_model_v2.pkl",
    "speech": "speech_model.pkl",
    "movement": "movement_model_final.pkl",
}

EXPECTED_POSITIVE = {
    "face": "affected",
    "speech": "dysarthria",
    "movement": "Incomplete",
}


def load_models():
    packages = {}

    for name, filename in MODEL_FILES.items():
        path = ROOT / "models" / filename
        package = joblib.load(path)

        if not isinstance(package, dict):
            raise TypeError(f"{name}: expected model package")

        required = {"model", "feature_columns", "label_map"}
        if not required.issubset(package):
            raise ValueError(f"{name}: missing package fields")

        model = package["model"]
        columns = package["feature_columns"]
        labels = package["label_map"]

        positive_name = EXPECTED_POSITIVE[name]
        positive_class = labels[positive_name]

        if not hasattr(model, "predict_proba"):
            raise ValueError(f"{name}: predict_proba missing")

        if positive_class not in model.classes_:
            raise ValueError(f"{name}: positive class missing")

        if len(columns) != model.n_features_in_:
            raise ValueError(f"{name}: feature count mismatch")

        packages[name] = package

        print(
            f"{name.upper()}: READY | "
            f"{len(columns)} features | "
            f"positive={positive_name}"
        )

    return packages


def predict_modality(package, features, positive_name):
    columns = package["feature_columns"]
    model = package["model"]
    positive_class = package["label_map"][positive_name]

    if not isinstance(features, dict):
        raise TypeError("Features must be a dictionary")

    missing = [col for col in columns if col not in features]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    values = np.array(
        [features[col] for col in columns],
        dtype=float,
    )

    if not np.all(np.isfinite(values)):
        raise ValueError("Features contain NaN or infinity")

    X = pd.DataFrame([values], columns=columns)

    probabilities = model.predict_proba(X)[0]
    class_index = list(model.classes_).index(positive_class)

    score = float(probabilities[class_index])

    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Invalid model probability")

    return score


if __name__ == "__main__":
    packages = load_models()
    print("\nModel loading and schema checks completed.")
    print("No predictions were made.")
