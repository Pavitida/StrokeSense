from pathlib import Path
import cv2
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

IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "face_landmark"
    / "test"
    / "test_face.jpg"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "face_landmark"
)

OUTPUT_PATH.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_IMAGE = (
    OUTPUT_PATH
    / "test_face_landmarks.jpg"
)


# =========================================================
# CHECK FILES
# =========================================================

print("=" * 70)
print("STEP 7 - TEST FACIAL LANDMARKS")
print("=" * 70)

print("\nModel:")
print(MODEL_PATH)

print("\nImage:")
print(IMAGE_PATH)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "face_landmarker.task not found"
    )

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        "test_face.jpg not found"
    )


# =========================================================
# MEDIAPIPE SETUP
# =========================================================

BaseOptions = mp.tasks.BaseOptions

FaceLandmarker = (
    mp.tasks.vision.FaceLandmarker
)

FaceLandmarkerOptions = (
    mp.tasks.vision.FaceLandmarkerOptions
)

RunningMode = (
    mp.tasks.vision.RunningMode
)


options = FaceLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH),
        delegate=BaseOptions.Delegate.CPU
    ),

    running_mode=RunningMode.IMAGE,

    num_faces=1,

    min_face_detection_confidence=0.5,

    min_face_presence_confidence=0.5,

    output_face_blendshapes=False,

    output_facial_transformation_matrixes=False
)


# =========================================================
# LOAD IMAGE
# =========================================================

image_bgr = cv2.imread(
    str(IMAGE_PATH)
)

if image_bgr is None:
    raise ValueError(
        "Could not read image."
    )


image_rgb = cv2.cvtColor(
    image_bgr,
    cv2.COLOR_BGR2RGB
)


mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=image_rgb
)


# =========================================================
# DETECT
# =========================================================

print("\nRunning Face Landmarker...")


with FaceLandmarker.create_from_options(
    options
) as landmarker:

    result = landmarker.detect(
        mp_image
    )


# =========================================================
# CHECK RESULT
# =========================================================

number_faces = len(
    result.face_landmarks
)

print(
    "\nFaces detected:",
    number_faces
)


if number_faces == 0:

    print("\nNO FACE DETECTED")

    print(
        "Try another clear front-facing image."
    )

    raise SystemExit


landmarks = (
    result.face_landmarks[0]
)

print(
    "Landmarks detected:",
    len(landmarks)
)


# =========================================================
# IMAGE SIZE
# =========================================================

height, width = (
    image_bgr.shape[:2]
)


# =========================================================
# DRAW ALL LANDMARKS
# =========================================================

output_image = (
    image_bgr.copy()
)


for landmark in landmarks:

    x = int(
        landmark.x * width
    )

    y = int(
        landmark.y * height
    )

    cv2.circle(
        output_image,
        (x, y),
        1,
        (0, 255, 0),
        -1
    )


# =========================================================
# IMPORTANT LANDMARKS
#
# MediaPipe landmark indices used here:
#
# mouth:
# 61  = one mouth corner
# 291 = opposite mouth corner
#
# eyes:
# 33  = outer corner
# 263 = opposite outer corner
#
# =========================================================

important_points = {

    "mouth_61": 61,
    "mouth_291": 291,

    "eye_33": 33,
    "eye_263": 263
}


print("\nIMPORTANT LANDMARKS")
print("-" * 50)


for name, index in important_points.items():

    landmark = landmarks[index]

    x = int(
        landmark.x * width
    )

    y = int(
        landmark.y * height
    )

    print(
        f"{name}: "
        f"x={landmark.x:.4f}, "
        f"y={landmark.y:.4f}, "
        f"z={landmark.z:.4f}"
    )

    cv2.circle(
        output_image,
        (x, y),
        5,
        (0, 0, 255),
        -1
    )

    cv2.putText(
        output_image,
        str(index),
        (x + 5, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1
    )


# =========================================================
# SAVE
# =========================================================

cv2.imwrite(
    str(OUTPUT_IMAGE),
    output_image
)


print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print(
    "\nOutput image:"
)

print(
    OUTPUT_IMAGE
)

print(
    "\nSTEP 7 COMPLETE"
)