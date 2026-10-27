from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "mediapipe"
    / "pose_landmarker.task"
)


print("=" * 70)
print("MOVEMENT STEP 2 - POSE LANDMARKER MODEL CHECK")
print("=" * 70)

print("\nMediaPipe version:")
print(mp.__version__)

print("\nModel:")
print(MODEL_PATH)

print("\nModel exists:")
print(MODEL_PATH.exists())

if not MODEL_PATH.exists():
    raise FileNotFoundError(MODEL_PATH)


# ============================================================
# CREATE POSE LANDMARKER
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=str(MODEL_PATH)
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1
)

print("\nCreating Pose Landmarker...")

with vision.PoseLandmarker.create_from_options(options):
    print("Pose Landmarker created successfully.")


print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print("Pose Landmarker: WORKING")

print("\nMOVEMENT STEP 2 COMPLETE")