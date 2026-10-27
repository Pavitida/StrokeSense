"""End-to-end experimental StrokeSense pipeline."""

import argparse
import json

from src.face.face_inference import predict_face
from src.speech.speech_inference import predict_speech
from src.movement.movement_inference import predict_movement


def run_pipeline(image_path, audio_path, video_path):
    face = predict_face(image_path)
    if face.get("status") != "success" or face.get("score") is None:
        return {
            "status": "incomplete",
            "failed_modality": "face",
            "reason": face.get("status", "Face analysis unavailable"),
            "fusion": None,
        }

    try:
        speech = predict_speech(audio_path)
    except ValueError as error:
        return {
            "status": "incomplete",
            "failed_modality": "speech",
            "reason": str(error),
            "fusion": None,
        }

    if speech.get("status") != "success" or speech.get("score") is None:
        return {
            "status": "incomplete",
            "failed_modality": "speech",
            "reason": speech.get("status", "Speech analysis unavailable"),
            "fusion": None,
        }

    try:
        movement = predict_movement(video_path)
    except ValueError as error:
        return {
            "status": "incomplete",
            "failed_modality": "movement",
            "reason": str(error),
            "fusion": None,
        }

    if movement.get("status") != "success" or movement.get("score") is None:
        return {
            "status": "incomplete",
            "failed_modality": "movement",
            "reason": movement.get("status", "Movement analysis unavailable"),
            "fusion": None,
        }

    scores = {
        "face": face["score"],
        "speech": speech["score"],
        "movement": movement["score"],
    }

    # Prototype aggregation only; not a calibrated medical risk score.
    combined_score = sum(scores.values()) / len(scores)

    return {
        "status": "success",
        "face": face,
        "speech": speech,
        "movement": movement,
        "fusion": {
            "method": "equal_weight_mean",
            "score": combined_score,
            "note": (
                "Experimental combined abnormality score. "
                "Not a stroke probability or medical diagnosis."
            ),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--video", required=True)
    args = parser.parse_args()

    result = run_pipeline(
        args.image,
        args.audio,
        args.video,
    )

    print(json.dumps(result, indent=2))
