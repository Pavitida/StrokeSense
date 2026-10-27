"""
StrokeSense - Prototype Score-Level Fusion

Combines three modality scores for software integration.
NOT a validated stroke probability or diagnostic model.
"""

from dataclasses import dataclass
from math import isfinite


@dataclass
class FusionResult:
    face_score: float
    speech_score: float
    movement_score: float
    combined_score: float
    method: str
    interpretation: str


def validate_score(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")

    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")

    return float(value)


def fuse_scores(face_score, speech_score, movement_score):
    face = validate_score("face_score", face_score)
    speech = validate_score("speech_score", speech_score)
    movement = validate_score("movement_score", movement_score)

    # Equal weights for prototype integration only
    combined = (face + speech + movement) / 3

    return FusionResult(
        face_score=face,
        speech_score=speech,
        movement_score=movement,
        combined_score=combined,
        method="Equal-weight score-level fusion (prototype)",
        interpretation=(
            "Experimental combined abnormality score. "
            "Not a stroke probability or medical diagnosis."
        ),
    )


if __name__ == "__main__":
    print("=== StrokeSense Fusion Test ===")

    # Synthetic scores for testing only
    result = fuse_scores(
        face_score=0.70,
        speech_score=0.60,
        movement_score=0.80,
    )

    print(f"Face Score: {result.face_score:.4f}")
    print(f"Speech Score: {result.speech_score:.4f}")
    print(f"Movement Score: {result.movement_score:.4f}")
    print(f"Combined Score: {result.combined_score:.4f}")
    print(f"Method: {result.method}")
    print(f"Note: {result.interpretation}")
