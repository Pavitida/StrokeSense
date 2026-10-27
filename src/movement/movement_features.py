"""Movement feature extraction for StrokeSense."""

import cv2
import numpy as np
import mediapipe as mp

TARGET_FPS = 10
VISIBILITY_THRESHOLD = 0.50

def get_point(landmarks, index):
    lm = landmarks[index]

    return np.array(
        [lm.x, lm.y, lm.z],
        dtype=np.float32
    )


def landmarks_visible(landmarks, indices):
    return all(
        landmarks[i].visibility >= VISIBILITY_THRESHOLD
        for i in indices
    )


def distance(a, b):
    return float(
        np.linalg.norm(a - b)
    )


def calculate_angle(a, b, c):
    """
    Calculate angle ABC in degrees.
    """

    ba = a - b
    bc = c - b

    denominator = (
        np.linalg.norm(ba)
        * np.linalg.norm(bc)
    )

    if denominator < 1e-8:
        return np.nan

    cosine = (
        np.dot(ba, bc)
        / denominator
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )

    return float(
        np.degrees(
            np.arccos(cosine)
        )
    )


def clean_values(values):
    values = np.asarray(
        values,
        dtype=float
    )

    return values[
        np.isfinite(values)
    ]


def safe_mean(values):
    values = clean_values(values)

    if len(values) == 0:
        return np.nan

    return float(
        np.mean(values)
    )


def safe_std(values):
    values = clean_values(values)

    if len(values) == 0:
        return np.nan

    return float(
        np.std(values)
    )


def safe_min(values):
    values = clean_values(values)

    if len(values) == 0:
        return np.nan

    return float(
        np.min(values)
    )


def safe_max(values):
    values = clean_values(values)

    if len(values) == 0:
        return np.nan

    return float(
        np.max(values)
    )


def safe_range(values):
    values = clean_values(values)

    if len(values) == 0:
        return np.nan

    return float(
        np.max(values)
        - np.min(values)
    )


def mean_absolute_diff(values):
    """
    Mean absolute frame-to-frame change.
    """

    values = clean_values(values)

    if len(values) < 2:
        return 0.0

    return float(
        np.mean(
            np.abs(
                np.diff(values)
            )
        )
    )


def process_video(video_path, landmarker):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        return None

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if fps <= 0:
        fps = 30.0

    # Sample approximately TARGET_FPS frames per second
    frame_step = max(
        1,
        int(round(fps / TARGET_FPS))
    )

    # --------------------------------------------------------
    # TEMPORAL MEASUREMENTS
    # --------------------------------------------------------

    left_elbow_angles = []
    right_elbow_angles = []

    left_shoulder_angles = []
    right_shoulder_angles = []

    left_arm_extensions = []
    right_arm_extensions = []

    left_wrist_heights = []
    right_wrist_heights = []

    elbow_angle_differences = []
    wrist_height_differences = []
    wrist_distances = []

    left_wrist_x = []
    left_wrist_y = []

    right_wrist_x = []
    right_wrist_y = []

    processed_frames = 0
    detected_frames = 0

    frame_index = 0

    required_landmarks = [
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        LEFT_ELBOW,
        RIGHT_ELBOW,
        LEFT_WRIST,
        RIGHT_WRIST,
        LEFT_HIP,
        RIGHT_HIP,
    ]

    # --------------------------------------------------------
    # READ VIDEO
    # --------------------------------------------------------

    while True:

        success, frame = cap.read()

        if not success:
            break

        # Skip frames according to sampling rate
        if frame_index % frame_step != 0:
            frame_index += 1
            continue

        processed_frames += 1

        # OpenCV BGR -> RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int(
            (frame_index / fps)
            * 1000
        )

        result = (
            landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )
        )

        # ----------------------------------------------------
        # POSE CHECK
        # ----------------------------------------------------

        if not result.pose_landmarks:
            frame_index += 1
            continue

        landmarks = (
            result.pose_landmarks[0]
        )

        # MediaPipe already detected a valid pose.
# Do not reject the entire frame just because one landmark
# has lower visibility. This dataset focuses on upper-limb
# rehabilitation and some body parts may be partially occluded.

        detected_frames += 1

        # ----------------------------------------------------
        # LANDMARK POINTS
        # ----------------------------------------------------

        left_shoulder = get_point(
            landmarks,
            LEFT_SHOULDER
        )

        right_shoulder = get_point(
            landmarks,
            RIGHT_SHOULDER
        )

        left_elbow = get_point(
            landmarks,
            LEFT_ELBOW
        )

        right_elbow = get_point(
            landmarks,
            RIGHT_ELBOW
        )

        left_wrist = get_point(
            landmarks,
            LEFT_WRIST
        )

        right_wrist = get_point(
            landmarks,
            RIGHT_WRIST
        )

        left_hip = get_point(
            landmarks,
            LEFT_HIP
        )

        right_hip = get_point(
            landmarks,
            RIGHT_HIP
        )

        # ----------------------------------------------------
        # ELBOW ANGLES
        #
        # shoulder -> elbow -> wrist
        # ----------------------------------------------------

        left_elbow_angle = calculate_angle(
            left_shoulder,
            left_elbow,
            left_wrist
        )

        right_elbow_angle = calculate_angle(
            right_shoulder,
            right_elbow,
            right_wrist
        )

        # ----------------------------------------------------
        # SHOULDER ANGLES
        #
        # hip -> shoulder -> elbow
        # ----------------------------------------------------

        left_shoulder_angle = calculate_angle(
            left_hip,
            left_shoulder,
            left_elbow
        )

        right_shoulder_angle = calculate_angle(
            right_hip,
            right_shoulder,
            right_elbow
        )

        # ----------------------------------------------------
        # ARM EXTENSION
        # ----------------------------------------------------

        left_arm_length = (
            distance(
                left_shoulder,
                left_elbow
            )
            +
            distance(
                left_elbow,
                left_wrist
            )
        )

        right_arm_length = (
            distance(
                right_shoulder,
                right_elbow
            )
            +
            distance(
                right_elbow,
                right_wrist
            )
        )

        if left_arm_length > 1e-8:

            left_extension = (
                distance(
                    left_shoulder,
                    left_wrist
                )
                / left_arm_length
            )

        else:
            left_extension = np.nan

        if right_arm_length > 1e-8:

            right_extension = (
                distance(
                    right_shoulder,
                    right_wrist
                )
                / right_arm_length
            )

        else:
            right_extension = np.nan

        # ----------------------------------------------------
        # WRIST HEIGHT RELATIVE TO SHOULDER
        #
        # MediaPipe y increases downward.
        # Positive value = wrist above shoulder.
        # ----------------------------------------------------

        left_wrist_height = (
            left_shoulder[1]
            - left_wrist[1]
        )

        right_wrist_height = (
            right_shoulder[1]
            - right_wrist[1]
        )

        # ----------------------------------------------------
        # SAVE FRAME FEATURES
        # ----------------------------------------------------

        left_elbow_angles.append(
            left_elbow_angle
        )

        right_elbow_angles.append(
            right_elbow_angle
        )

        left_shoulder_angles.append(
            left_shoulder_angle
        )

        right_shoulder_angles.append(
            right_shoulder_angle
        )

        left_arm_extensions.append(
            left_extension
        )

        right_arm_extensions.append(
            right_extension
        )

        left_wrist_heights.append(
            left_wrist_height
        )

        right_wrist_heights.append(
            right_wrist_height
        )

        elbow_angle_differences.append(
            abs(
                left_elbow_angle
                - right_elbow_angle
            )
        )

        wrist_height_differences.append(
            abs(
                left_wrist_height
                - right_wrist_height
            )
        )

        wrist_distances.append(
            distance(
                left_wrist,
                right_wrist
            )
        )

        left_wrist_x.append(
            left_wrist[0]
        )

        left_wrist_y.append(
            left_wrist[1]
        )

        right_wrist_x.append(
            right_wrist[0]
        )

        right_wrist_y.append(
            right_wrist[1]
        )

        frame_index += 1

    cap.release()

    # --------------------------------------------------------
    # NO VALID POSE
    # --------------------------------------------------------

    if detected_frames == 0:
        return None

    pose_detection_rate = (
        detected_frames
        / max(
            processed_frames,
            1
        )
    )

    # ========================================================
    # VIDEO-LEVEL FEATURES
    # ========================================================

    features = {

        # ----------------------------------------------------
        # VIDEO INFORMATION
        # ----------------------------------------------------

        "video_fps":
            float(fps),

        "video_total_frames":
            int(total_frames),

        "processed_frames":
            int(processed_frames),

        "detected_frames":
            int(detected_frames),

        "pose_detection_rate":
            float(pose_detection_rate),

        # ----------------------------------------------------
        # LEFT ELBOW
        # ----------------------------------------------------

        "left_elbow_mean":
            safe_mean(
                left_elbow_angles
            ),

        "left_elbow_std":
            safe_std(
                left_elbow_angles
            ),

        "left_elbow_min":
            safe_min(
                left_elbow_angles
            ),

        "left_elbow_max":
            safe_max(
                left_elbow_angles
            ),

        "left_elbow_rom":
            safe_range(
                left_elbow_angles
            ),

        # ----------------------------------------------------
        # RIGHT ELBOW
        # ----------------------------------------------------

        "right_elbow_mean":
            safe_mean(
                right_elbow_angles
            ),

        "right_elbow_std":
            safe_std(
                right_elbow_angles
            ),

        "right_elbow_min":
            safe_min(
                right_elbow_angles
            ),

        "right_elbow_max":
            safe_max(
                right_elbow_angles
            ),

        "right_elbow_rom":
            safe_range(
                right_elbow_angles
            ),

        # ----------------------------------------------------
        # ELBOW ASYMMETRY
        # ----------------------------------------------------

        "elbow_asymmetry_mean":
            safe_mean(
                elbow_angle_differences
            ),

        "elbow_asymmetry_std":
            safe_std(
                elbow_angle_differences
            ),

        # ----------------------------------------------------
        # LEFT SHOULDER
        # ----------------------------------------------------

        "left_shoulder_mean":
            safe_mean(
                left_shoulder_angles
            ),

        "left_shoulder_std":
            safe_std(
                left_shoulder_angles
            ),

        "left_shoulder_min":
            safe_min(
                left_shoulder_angles
            ),

        "left_shoulder_max":
            safe_max(
                left_shoulder_angles
            ),

        "left_shoulder_rom":
            safe_range(
                left_shoulder_angles
            ),

        # ----------------------------------------------------
        # RIGHT SHOULDER
        # ----------------------------------------------------

        "right_shoulder_mean":
            safe_mean(
                right_shoulder_angles
            ),

        "right_shoulder_std":
            safe_std(
                right_shoulder_angles
            ),

        "right_shoulder_min":
            safe_min(
                right_shoulder_angles
            ),

        "right_shoulder_max":
            safe_max(
                right_shoulder_angles
            ),

        "right_shoulder_rom":
            safe_range(
                right_shoulder_angles
            ),

        # ----------------------------------------------------
        # LEFT ARM EXTENSION
        # ----------------------------------------------------

        "left_extension_mean":
            safe_mean(
                left_arm_extensions
            ),

        "left_extension_std":
            safe_std(
                left_arm_extensions
            ),

        "left_extension_min":
            safe_min(
                left_arm_extensions
            ),

        "left_extension_max":
            safe_max(
                left_arm_extensions
            ),

        "left_extension_rom":
            safe_range(
                left_arm_extensions
            ),

        # ----------------------------------------------------
        # RIGHT ARM EXTENSION
        # ----------------------------------------------------

        "right_extension_mean":
            safe_mean(
                right_arm_extensions
            ),

        "right_extension_std":
            safe_std(
                right_arm_extensions
            ),

        "right_extension_min":
            safe_min(
                right_arm_extensions
            ),

        "right_extension_max":
            safe_max(
                right_arm_extensions
            ),

        "right_extension_rom":
            safe_range(
                right_arm_extensions
            ),

        # ----------------------------------------------------
        # LEFT WRIST HEIGHT
        # ----------------------------------------------------

        "left_wrist_height_mean":
            safe_mean(
                left_wrist_heights
            ),

        "left_wrist_height_std":
            safe_std(
                left_wrist_heights
            ),

        "left_wrist_height_min":
            safe_min(
                left_wrist_heights
            ),

        "left_wrist_height_max":
            safe_max(
                left_wrist_heights
            ),

        "left_wrist_height_rom":
            safe_range(
                left_wrist_heights
            ),

        # ----------------------------------------------------
        # RIGHT WRIST HEIGHT
        # ----------------------------------------------------

        "right_wrist_height_mean":
            safe_mean(
                right_wrist_heights
            ),

        "right_wrist_height_std":
            safe_std(
                right_wrist_heights
            ),

        "right_wrist_height_min":
            safe_min(
                right_wrist_heights
            ),

        "right_wrist_height_max":
            safe_max(
                right_wrist_heights
            ),

        "right_wrist_height_rom":
            safe_range(
                right_wrist_heights
            ),

        # ----------------------------------------------------
        # WRIST ASYMMETRY / DISTANCE
        # ----------------------------------------------------

        "wrist_height_asymmetry_mean":
            safe_mean(
                wrist_height_differences
            ),

        "wrist_height_asymmetry_std":
            safe_std(
                wrist_height_differences
            ),

        "wrist_distance_mean":
            safe_mean(
                wrist_distances
            ),

        "wrist_distance_std":
            safe_std(
                wrist_distances
            ),

        "wrist_distance_rom":
            safe_range(
                wrist_distances
            ),

        # ----------------------------------------------------
        # TEMPORAL WRIST MOVEMENT
        # ----------------------------------------------------

        "left_wrist_x_motion":
            mean_absolute_diff(
                left_wrist_x
            ),

        "left_wrist_y_motion":
            mean_absolute_diff(
                left_wrist_y
            ),

        "right_wrist_x_motion":
            mean_absolute_diff(
                right_wrist_x
            ),

        "right_wrist_y_motion":
            mean_absolute_diff(
                right_wrist_y
            ),

        # ----------------------------------------------------
        # TEMPORAL JOINT MOVEMENT
        # ----------------------------------------------------

        "left_elbow_motion":
            mean_absolute_diff(
                left_elbow_angles
            ),

        "right_elbow_motion":
            mean_absolute_diff(
                right_elbow_angles
            ),

        "left_shoulder_motion":
            mean_absolute_diff(
                left_shoulder_angles
            ),

        "right_shoulder_motion":
            mean_absolute_diff(
                right_shoulder_angles
            ),
    }

    return features


# Landmark indices from original training code
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
