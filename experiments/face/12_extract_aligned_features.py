from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "mediapipe"
    / "face_landmarker.task"
)

INPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "face_landmark"
    / "palsynet_split.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "face_landmark"
    / "palsynet_aligned_features.csv"
)


# ============================================================
# MEDIAPIPE
# ============================================================

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


# ============================================================
# HELPERS
# ============================================================

def p(landmarks, index):
    lm = landmarks[index]

    return np.array(
        [lm.x, lm.y],
        dtype=np.float64
    )


def dist(a, b):
    return float(
        np.linalg.norm(a - b)
    )


def rotate_points(points, center, angle):

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    rotation = np.array(
        [
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ]
    )

    return (
        points - center
    ) @ rotation.T + center


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_aligned_features(landmarks):

    # --------------------------------------------------------
    # Convert all 478 landmarks to XY
    # --------------------------------------------------------

    pts = np.array(
        [
            [lm.x, lm.y]
            for lm in landmarks
        ],
        dtype=np.float64
    )


    # --------------------------------------------------------
    # FACE ALIGNMENT
    #
    # 33  = one outer eye corner
    # 263 = opposite outer eye corner
    # --------------------------------------------------------

    eye_left_outer = pts[33]
    eye_right_outer = pts[263]

    eye_vector = (
        eye_right_outer
        - eye_left_outer
    )

    angle = np.arctan2(
        eye_vector[1],
        eye_vector[0]
    )

    eye_center = (
        eye_left_outer
        + eye_right_outer
    ) / 2.0


    # Rotate in the opposite direction
    # so eye line becomes horizontal.

    aligned = rotate_points(
        pts,
        eye_center,
        -angle
    )


    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    scale = dist(
        aligned[33],
        aligned[263]
    )

    if scale < 1e-8:
        return None

    aligned = (
        aligned - eye_center
    ) / scale


    # convenient helper

    def q(index):
        return aligned[index]


    # ========================================================
    # EYES
    # ========================================================

    # Left eye vertical distances

    left_eye_open_1 = dist(
        q(159),
        q(145)
    )

    left_eye_open_2 = dist(
        q(160),
        q(144)
    )


    # Right eye vertical distances

    right_eye_open_1 = dist(
        q(386),
        q(374)
    )

    right_eye_open_2 = dist(
        q(385),
        q(380)
    )


    left_eye_open = np.mean(
        [
            left_eye_open_1,
            left_eye_open_2
        ]
    )

    right_eye_open = np.mean(
        [
            right_eye_open_1,
            right_eye_open_2
        ]
    )


    eye_open_diff = abs(
        left_eye_open
        - right_eye_open
    )


    eye_open_ratio = (
        min(
            left_eye_open,
            right_eye_open
        )
        /
        (
            max(
                left_eye_open,
                right_eye_open
            )
            + 1e-8
        )
    )


    # ========================================================
    # EYEBROWS
    # ========================================================

    left_brow_points = [
        70, 63, 105, 66, 107
    ]

    right_brow_points = [
        336, 296, 334, 293, 300
    ]


    left_brow_y = np.mean(
        [
            q(i)[1]
            for i in left_brow_points
        ]
    )

    right_brow_y = np.mean(
        [
            q(i)[1]
            for i in right_brow_points
        ]
    )


    # Eye reference levels

    left_eye_y = np.mean(
        [
            q(33)[1],
            q(133)[1]
        ]
    )

    right_eye_y = np.mean(
        [
            q(263)[1],
            q(362)[1]
        ]
    )


    left_brow_height = abs(
        left_brow_y
        - left_eye_y
    )

    right_brow_height = abs(
        right_brow_y
        - right_eye_y
    )


    brow_height_diff = abs(
        left_brow_height
        - right_brow_height
    )


    brow_height_ratio = (
        min(
            left_brow_height,
            right_brow_height
        )
        /
        (
            max(
                left_brow_height,
                right_brow_height
            )
            + 1e-8
        )
    )


    # ========================================================
    # MOUTH
    # ========================================================

    mouth_left = q(61)
    mouth_right = q(291)

    upper_lip = q(13)
    lower_lip = q(14)

    nose = q(1)


    mouth_width = dist(
        mouth_left,
        mouth_right
    )


    mouth_open = dist(
        upper_lip,
        lower_lip
    )


    # Vertical difference after head alignment

    mouth_corner_y_diff = abs(
        mouth_left[1]
        - mouth_right[1]
    )


    # Distance from nose to each mouth corner

    left_mouth_nose = dist(
        nose,
        mouth_left
    )

    right_mouth_nose = dist(
        nose,
        mouth_right
    )


    mouth_nose_diff = abs(
        left_mouth_nose
        - right_mouth_nose
    )


    mouth_nose_ratio = (
        min(
            left_mouth_nose,
            right_mouth_nose
        )
        /
        (
            max(
                left_mouth_nose,
                right_mouth_nose
            )
            + 1e-8
        )
    )


    # ========================================================
    # CHEEKS
    # ========================================================

    # Representative left/right cheek points

    left_cheek = q(234)
    right_cheek = q(454)


    left_cheek_nose = dist(
        nose,
        left_cheek
    )

    right_cheek_nose = dist(
        nose,
        right_cheek
    )


    cheek_diff = abs(
        left_cheek_nose
        - right_cheek_nose
    )


    cheek_ratio = (
        min(
            left_cheek_nose,
            right_cheek_nose
        )
        /
        (
            max(
                left_cheek_nose,
                right_cheek_nose
            )
            + 1e-8
        )
    )


    # ========================================================
    # NOSE / FACE CENTER OFFSET
    # ========================================================

    face_left = q(234)
    face_right = q(454)

    face_center_x = (
        face_left[0]
        + face_right[0]
    ) / 2.0


    nose_center_offset = abs(
        nose[0]
        - face_center_x
    )


    # ========================================================
    # SYMMETRIC PAIR FEATURES
    # ========================================================

    # These pairs represent corresponding regions
    # on opposite sides of the face.

    symmetric_pairs = [

        # eye / surrounding area
        (33, 263),
        (133, 362),
        (159, 386),
        (145, 374),

        # eyebrow
        (70, 300),
        (63, 293),
        (105, 334),
        (66, 296),
        (107, 336),

        # mouth
        (61, 291),
        (78, 308),
        (95, 324),
        (88, 318),

        # cheeks / outer face
        (234, 454),
        (93, 323),
        (132, 361)
    ]


    midpoint_x = 0.0

    pair_errors = []


    for left_index, right_index in symmetric_pairs:

        left = q(left_index)
        right = q(right_index)


        # Mirror left point across vertical midline.

        mirrored_left = np.array(
            [
                2 * midpoint_x
                - left[0],

                left[1]
            ]
        )


        error = dist(
            mirrored_left,
            right
        )

        pair_errors.append(
            error
        )


    pair_errors = np.array(
        pair_errors
    )


    symmetry_mean = float(
        np.mean(pair_errors)
    )

    symmetry_median = float(
        np.median(pair_errors)
    )

    symmetry_max = float(
        np.max(pair_errors)
    )

    symmetry_std = float(
        np.std(pair_errors)
    )


    # ========================================================
    # REGIONAL ASYMMETRY
    # ========================================================

    eye_region = pair_errors[0:4]
    brow_region = pair_errors[4:9]
    mouth_region = pair_errors[9:13]
    cheek_region = pair_errors[13:16]


    eye_region_asymmetry = float(
        np.mean(eye_region)
    )

    brow_region_asymmetry = float(
        np.mean(brow_region)
    )

    mouth_region_asymmetry = float(
        np.mean(mouth_region)
    )

    cheek_region_asymmetry = float(
        np.mean(cheek_region)
    )


    # ========================================================
    # OVERALL SCORE
    # ========================================================

    overall_asymmetry = float(
        np.mean(
            [
                eye_open_diff,
                brow_height_diff,
                mouth_corner_y_diff,
                mouth_nose_diff,
                cheek_diff,
                symmetry_mean
            ]
        )
    )


    # ========================================================
    # RETURN
    # ========================================================

    return {

        "left_eye_open":
            left_eye_open,

        "right_eye_open":
            right_eye_open,

        "eye_open_diff":
            eye_open_diff,

        "eye_open_ratio":
            eye_open_ratio,


        "left_brow_height":
            left_brow_height,

        "right_brow_height":
            right_brow_height,

        "brow_height_diff":
            brow_height_diff,

        "brow_height_ratio":
            brow_height_ratio,


        "mouth_width":
            mouth_width,

        "mouth_open":
            mouth_open,

        "mouth_corner_y_diff":
            mouth_corner_y_diff,

        "left_mouth_nose":
            left_mouth_nose,

        "right_mouth_nose":
            right_mouth_nose,

        "mouth_nose_diff":
            mouth_nose_diff,

        "mouth_nose_ratio":
            mouth_nose_ratio,


        "left_cheek_nose":
            left_cheek_nose,

        "right_cheek_nose":
            right_cheek_nose,

        "cheek_diff":
            cheek_diff,

        "cheek_ratio":
            cheek_ratio,


        "nose_center_offset":
            nose_center_offset,


        "symmetry_mean":
            symmetry_mean,

        "symmetry_median":
            symmetry_median,

        "symmetry_max":
            symmetry_max,

        "symmetry_std":
            symmetry_std,


        "eye_region_asymmetry":
            eye_region_asymmetry,

        "brow_region_asymmetry":
            brow_region_asymmetry,

        "mouth_region_asymmetry":
            mouth_region_asymmetry,

        "cheek_region_asymmetry":
            cheek_region_asymmetry,


        "overall_asymmetry":
            overall_asymmetry
    }


# ============================================================
# LOAD EXISTING SPLIT
# ============================================================

print("=" * 70)
print("STEP 12A - EXTRACT ALIGNED FACIAL FEATURES")
print("=" * 70)

df = pd.read_csv(INPUT_PATH)


print("\nFrames:")
print(len(df))

print("\nPeople:")
print(
    df["person_id"].nunique()
)

print("\nSplit:")
print(
    df["split"].value_counts()
)


# ============================================================
# PROCESS
# ============================================================

rows = []

failed_detection = 0
failed_read = 0


with FaceLandmarker.create_from_options(
    options
) as landmarker:

    for i, row in df.iterrows():

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

            image_format=
                mp.ImageFormat.SRGB,

            data=image_rgb
        )


        result = landmarker.detect(
            mp_image
        )


        if len(
            result.face_landmarks
        ) == 0:

            failed_detection += 1
            continue


        landmarks = (
            result.face_landmarks[0]
        )


        features = (
            extract_aligned_features(
                landmarks
            )
        )


        if features is None:

            failed_detection += 1
            continue


        output_row = {

            "person_id":
                row["person_id"],

            "label":
                row["label"],

            "split":
                row["split"],

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
                f"{i + 1}/{len(df)}"
            )


# ============================================================
# SAVE
# ============================================================

features_df = pd.DataFrame(
    rows
)

features_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 12A RESULT")
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

    print(
        "\nPeople:",
        features_df[
            "person_id"
        ].nunique()
    )

    print("\nSplit:")

    print(
        features_df[
            "split"
        ].value_counts()
    )


    metadata_columns = [

        "person_id",
        "label",
        "split",
        "video_file",
        "frame_number",
        "image_path"
    ]


    feature_columns = [

        c
        for c in features_df.columns
        if c not in metadata_columns
    ]


    print(
        "\nFeature count:",
        len(feature_columns)
    )

    print("\nFeatures:")

    for feature in feature_columns:
        print("•", feature)


    # ---------------------------------------------
    # Check NaN / infinity
    # ---------------------------------------------

    feature_values = (
        features_df[
            feature_columns
        ]
    )


    nan_count = int(
        feature_values
        .isna()
        .sum()
        .sum()
    )


    inf_count = int(
        np.isinf(
            feature_values.values
        ).sum()
    )


    print(
        "\nNaN values:",
        nan_count
    )

    print(
        "Infinite values:",
        inf_count
    )


print("\nSaved:")
print(OUTPUT_PATH)

print("\nSTEP 12A COMPLETE")