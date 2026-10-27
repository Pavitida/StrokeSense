from pathlib import Path
import mediapipe as mp


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "mediapipe"
    / "face_landmarker.task"
)

print("=" * 60)
print("STEP 6B - MEDIAPIPE FACE LANDMARKER")
print("=" * 60)

print("\nMediaPipe version:")
print(mp.__version__)

print("\nModel path:")
print(MODEL_PATH)

print("\nModel exists:")
print(MODEL_PATH.exists())


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "face_landmarker.task not found."
    )


# =========================================================
# CREATE FACE LANDMARKER
# =========================================================

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH),
        delegate=BaseOptions.Delegate.CPU
    ),

    running_mode=VisionRunningMode.IMAGE,

    num_faces=1,

    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,

    # ตอนนี้ปิดก่อน
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)