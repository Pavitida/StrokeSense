from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "mediapipe"
    / "pose_landmarker.task"
)

IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "movement"
    / "test"
    / "test_pose.jpg"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "movement"
    / "test_pose_landmarks.jpg"
)


print("=" * 70)
print("MOVEMENT STEP 3 - TEST POSE LANDMARKS")
print("=" * 70)

print("\nModel:")
print(MODEL_PATH)

print("\nImage:")
print(IMAGE_PATH)

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        "test_pose.jpg not found"
    )


# ============================================================
# CREATE LANDMARKER
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=str(MODEL_PATH)
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1
)


# ============================================================
# LOAD IMAGE
# ============================================================

mp_image = mp.Image.create_from_file(
    str(IMAGE_PATH)
)


# ============================================================
# DETECT
# ============================================================

print("\nRunning Pose Landmarker...")

with vision.PoseLandmarker.create_from_options(
    options
) as landmarker:

    result = landmarker.detect(mp_image)


print("\nPoses detected:")
print(len(result.pose_landmarks))

if len(result.pose_landmarks) == 0:
    raise RuntimeError(
        "No pose detected."
    )

landmarks = result.pose_landmarks[0]

print("\nLandmarks detected:")
print(len(landmarks))


# ============================================================
# IMPORTANT ARM LANDMARKS
# ============================================================

important = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
}


print("\n" + "=" * 70)
print("IMPORTANT ARM LANDMARKS")
print("=" * 70)

for name, index in important.items():

    lm = landmarks[index]

    print(
        f"{name:16s} "
        f"x={lm.x:.4f}, "
        f"y={lm.y:.4f}, "
        f"z={lm.z:.4f}, "
        f"visibility={lm.visibility:.4f}"
    )


# ============================================================
# DRAW IMPORTANT POINTS
# ============================================================

image = cv2.imread(
    str(IMAGE_PATH)
)

if image is None:
    raise RuntimeError(
        "Could not read image with OpenCV."
    )

height, width = image.shape[:2]


# Connections for arms
connections = [
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 12),
]


for start, end in connections:

    p1 = landmarks[start]
    p2 = landmarks[end]

    x1 = int(p1.x * width)
    y1 = int(p1.y * height)

    x2 = int(p2.x * width)
    y2 = int(p2.y * height)

    cv2.line(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3
    )


for name, index in important.items():

    lm = landmarks[index]

    x = int(lm.x * width)
    y = int(lm.y * height)

    cv2.circle(
        image,
        (x, y),
        7,
        (0, 0, 255),
        -1
    )

    cv2.putText(
        image,
        name,
        (x + 8, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        2
    )


OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

cv2.imwrite(
    str(OUTPUT_PATH),
    image
)


# ============================================================
# RESULT
# ============================================================

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print("\nOutput image:")
print(OUTPUT_PATH)

print("\nMOVEMENT STEP 3 COMPLETE")