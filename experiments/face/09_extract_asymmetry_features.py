from pathlib import Path
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np


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

METADATA_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "face_landmark"
    / "palsynet_frames_metadata.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "face_landmark"
    / "palsynet_asymmetry_features.csv"
)


# =========================================================
# MEDIAPIPE
# =========================================================

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


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
# HELPER FUNCTIONS
# =========================================================

def point(landmarks, index):
    """Return x,y coordinates."""
    lm = landmarks[index]

    return np.array(
        [lm.x, lm.y],
        dtype=np.float32
    )


def distance(a, b):
    return float(
        np.linalg.norm(a - b)
    )


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_features(landmarks):

    # -----------------------------------------------------
    # IMPORTANT LANDMARKS
    # -----------------------------------------------------

    # Eye outer corners
    left_eye_outer = point(landmarks, 33)
    right_eye_outer = point(landmarks, 263)

    # Eye inner corners
    left_eye_inner = point(landmarks, 133)
    right_eye_inner = point(landmarks, 362)

    # Eye upper/lower points
    left_eye_top = point(landmarks, 159)
    left_eye_bottom = point(landmarks, 145)

    right_eye_top = point(landmarks, 386)
    right_eye_bottom = point(landmarks, 374)

    # Eyebrow points
    left_brow = point(landmarks, 105)
    right_brow = point(landmarks, 334)

    # Mouth corners
    mouth_left = point(landmarks, 61)
    mouth_right = point(landmarks, 291)

    # Upper / lower lip
    upper_lip = point(landmarks, 13)
    lower_lip = point(landmarks, 14)

    # Nose / central reference
    nose = point(landmarks, 1)

    # Chin
    chin = point(landmarks, 152)


    # -----------------------------------------------------
    # NORMALIZATION
    # -----------------------------------------------------

    # Normalize all distances by inter-eye distance.
    inter_eye_distance = distance(
        left_eye_outer,
        right_eye_outer
    )

    if inter_eye_distance < 1e-6:
        return None


    # -----------------------------------------------------
    # EYE FEATURES
    # -----------------------------------------------------

    left_eye_open = (
        distance(
            left_eye_top,
            left_eye_bottom
        )
        / inter_eye_distance
    )

    right_eye_open = (
        distance(
            right_eye_top,
            right_eye_bottom
        )
        / inter_eye_distance
    )

    eye_open_diff = abs(
        left_eye_open
        - right_eye_open
    )


    # -----------------------------------------------------
    # EYEBROW FEATURES
    # -----------------------------------------------------

    left_brow_height = abs(
        left_brow[1]
        - left_eye_top[1]
    ) / inter_eye_distance

    right_brow_height = abs(
        right_brow[1]
        - right_eye_top[1]
    ) / inter_eye_distance

    eyebrow_height_diff = abs(
        left_brow_height
        - right_brow_height
    )


    # -----------------------------------------------------
    # MOUTH FEATURES
    # -----------------------------------------------------

    mouth_width = (
        distance(
            mouth_left,
            mouth_right
        )
        / inter_eye_distance
    )

    mouth_open = (
        distance(
            upper_lip,
            lower_lip
        )
        / inter_eye_distance
    )

    mouth_height_diff = (
        abs(
            mouth_left[1]
            - mouth_right[1]
        )
        / inter_eye_distance
    )

    mouth_slope = (
        abs(
            mouth_right[1]
            - mouth_left[1]
        )
        /
        (
            abs(
                mouth_right[0]
                - mouth_left[0]
            )
            + 1e-6
        )
    )


    # -----------------------------------------------------
    # MOUTH LEFT / RIGHT DISTANCE FROM NOSE
    # -----------------------------------------------------

    mouth_left_nose = (
        distance(
            mouth_left,
            nose
        )
        / inter_eye_distance
    )

    mouth_right_nose = (
        distance(
            mouth_right,
            nose
        )
        / inter_eye_distance
    )

    mouth_nose_diff = abs(
        mouth_left_nose
        - mouth_right_nose
    )


    # -----------------------------------------------------
    # EYE LEFT / RIGHT DISTANCE FROM NOSE
    # -----------------------------------------------------

    left_eye_nose = (
        distance(
            left_eye_inner,
            nose
        )
        / inter_eye_distance
    )

    right_eye_nose = (
        distance(
            right_eye_inner,
            nose
        )
        / inter_eye_distance
    )

    eye_nose_diff = abs(
        left_eye_nose
        - right_eye_nose
    )


    # -----------------------------------------------------
    # FACE LENGTH
    # -----------------------------------------------------

    face_length = (
        distance(
            nose,
            chin
        )
        / inter_eye_distance
    )


    # -----------------------------------------------------
    # OVERALL ASYMMETRY
    # -----------------------------------------------------

    overall_asymmetry = np.mean(
        [
            eye_open_diff,
            eyebrow_height_diff,
            mouth_height_diff,
            mouth_nose_diff,
            eye_nose_diff
        ]
    )


    return {

        "left_eye_open": left_eye_open,
        "right_eye_open": right_eye_open,
        "eye_open_diff": eye_open_diff,

        "left_brow_height": left_brow_height,
        "right_brow_height": right_brow_height,
        "eyebrow_height_diff": eyebrow_height_diff,

        "mouth_width": mouth_width,
        "mouth_open": mouth_open,
        "mouth_height_diff": mouth_height_diff,
        "mouth_slope": mouth_slope,

        "mouth_left_nose": mouth_left_nose,
        "mouth_right_nose": mouth_right_nose,
        "mouth_nose_diff": mouth_nose_diff,

        "left_eye_nose": left_eye_nose,
        "right_eye_nose": right_eye_nose,
        "eye_nose_diff": eye_nose_diff,

        "face_length": face_length,

        "overall_asymmetry": overall_asymmetry
    }


# =========================================================
# LOAD METADATA
# =========================================================

print("=" * 70)
print("STEP 9 - EXTRACT FACIAL ASYMMETRY FEATURES")
print("=" * 70)

metadata = pd.read_csv(
    METADATA_PATH
)

print("\nImages:")
print(len(metadata))

print("\nPeople:")
print(
    metadata["person_id"].nunique()
)

print("\nLabels:")
print(
    metadata["label"].value_counts()
)


# =========================================================
# PROCESS
# =========================================================

rows = []

failed_detection = 0
failed_read = 0


with FaceLandmarker.create_from_options(
    options
) as landmarker:

    for i, row in metadata.iterrows():

        image = cv2.imread(
            row["image_path"]
        )

        if image is None:
            failed_read += 1
            continue


        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_rgb
        )


        result = landmarker.detect(
            mp_image
        )


        if len(result.face_landmarks) == 0:

            failed_detection += 1
            continue


        landmarks = (
            result.face_landmarks[0]
        )


        features = extract_features(
            landmarks
        )


        if features is None:
            failed_detection += 1
            continue


        output_row = {

            "person_id":
                row["person_id"],

            "label":
                row["label"],

            "video_file":
                row["video_file"],

            "frame_number":
                row["frame_number"],

            "image_path":
                row["image_path"]
        }


        output_row.update(
            features
        )


        rows.append(
            output_row
        )


        if (i + 1) % 100 == 0:

            print(
                f"Processed "
                f"{i + 1}/{len(metadata)}"
            )


# =========================================================
# SAVE
# =========================================================

features_df = pd.DataFrame(
    rows
)

features_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# =========================================================
# SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("FEATURE EXTRACTION RESULT")
print("=" * 70)

print(
    "\nSuccessful:",
    len(features_df)
)

print(
    "Failed face detection:",
    failed_detection
)

print(
    "Failed image read:",
    failed_read
)


if len(features_df) > 0:

    success_rate = (
        len(features_df)
        / len(metadata)
        * 100
    )

    print(
        f"Success rate: "
        f"{success_rate:.2f}%"
    )


    print("\nImages by label:")

    print(
        features_df[
            "label"
        ].value_counts()
    )


    print("\nPeople by label:")

    print(
        features_df.groupby(
            "label"
        )["person_id"].nunique()
    )


    print("\nFeature columns:")

    feature_columns = [
        c
        for c in features_df.columns
        if c not in [
            "person_id",
            "label",
            "video_file",
            "frame_number",
            "image_path"
        ]
    ]

    print(
        len(feature_columns)
    )

    print(
        feature_columns
    )


print("\nSaved:")
print(OUTPUT_PATH)

print("\n" + "=" * 70)
print("STEP 9 COMPLETE")
print("=" * 70)