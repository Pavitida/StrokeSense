"""Experimental speech inference for StrokeSense."""

from pathlib import Path
import argparse

import joblib
import librosa
import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "speech_model.pkl"


def extract_speech_features(audio_path, target_sr=16000, n_mfcc=13):
    audio_path = Path(audio_path)

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    y, sr = sf.read(str(audio_path), dtype="float32")

    if y.ndim > 1:
        y = np.mean(y, axis=1)

    if sr != target_sr:
        y = librosa.resample(
            y,
            orig_sr=sr,
            target_sr=target_sr,
        )
        sr = target_sr

    if len(y) == 0:
        raise ValueError("Empty audio")

    if not np.isfinite(y).all():
        raise ValueError("Audio contains invalid values")

    peak = np.max(np.abs(y))

    if peak <= 1e-8:
        raise ValueError("Audio is silent")

    y = y / peak

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    features = {}

    for i in range(n_mfcc):
        features[f"mfcc_{i + 1}_mean"] = float(mfcc_mean[i])
        features[f"mfcc_{i + 1}_std"] = float(mfcc_std[i])

    return features


def predict_speech(audio_path):
    package = joblib.load(MODEL_PATH)

    model = package["model"]
    columns = package["feature_columns"]
    label_map = package["label_map"]

    target_sr = package.get("target_sample_rate", 16000)
    n_mfcc = package.get("n_mfcc", 13)

    features = extract_speech_features(
        audio_path,
        target_sr=target_sr,
        n_mfcc=n_mfcc,
    )

    missing = [name for name in columns if name not in features]

    if missing:
        raise ValueError(f"Missing features: {missing}")

    values = np.array(
        [[features[name] for name in columns]],
        dtype=np.float64,
    )

    if not np.isfinite(values).all():
        raise ValueError("Features contain NaN or infinity")

    positive_class = label_map["dysarthria"]
    classes = list(model.classes_)

    if positive_class not in classes:
        raise ValueError("Positive class not found")

    index = classes.index(positive_class)
    score = float(model.predict_proba(values)[0][index])

    return {
        "status": "success",
        "score": score,
        "feature_count": len(columns),
        "positive_class": "dysarthria",
        "note": (
            "Experimental dysarthria classification score; "
            "not a stroke diagnosis or stroke probability."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    args = parser.parse_args()

    print(predict_speech(args.audio))
