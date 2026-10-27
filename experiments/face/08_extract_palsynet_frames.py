from pathlib import Path
import cv2
import pandas as pd


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "palsynet"
    / "data"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "palsynet_frames"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "face_landmark"
    / "palsynet_frames_metadata.csv"
)

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)


# =========================================================
# SETTINGS
# =========================================================

# Sample approximately one frame every second.
SAMPLE_SECONDS = 1.0

# Prevent very long videos from dominating the dataset.
MAX_FRAMES_PER_PERSON = 100

LABELS = [
    "affected",
    "unaffected"
]


# =========================================================
# START
# =========================================================

print("=" * 70)
print("STEP 8B - EXTRACT PALS YNET FRAMES")
print("=" * 70)

print("\nDataset:")
print(DATASET_PATH)

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}"
    )


metadata = []

total_videos = 0
total_saved = 0
failed_videos = []


# =========================================================
# PROCESS EACH CLASS
# =========================================================

for label in LABELS:

    class_path = DATASET_PATH / label

    if not class_path.exists():
        print(f"\nWARNING: {class_path} does not exist.")
        continue

    videos = sorted(
        class_path.glob("*.mp4")
    )

    print("\n" + "=" * 70)
    print(label.upper())
    print("=" * 70)

    print("Videos:", len(videos))

    for video_path in videos:

        total_videos += 1

        # Example:
        # affected_7
        person_id = (
            f"{label}_{video_path.stem}"
        )

        person_output = (
            OUTPUT_PATH
            / label
            / person_id
        )

        person_output.mkdir(
            parents=True,
            exist_ok=True
        )

        cap = cv2.VideoCapture(
            str(video_path)
        )

        if not cap.isOpened():

            print(
                f"FAILED: {video_path.name}"
            )

            failed_videos.append(
                str(video_path)
            )

            continue


        # ---------------------------------------------
        # VIDEO INFORMATION
        # ---------------------------------------------

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        duration = (
            total_frames / fps
            if fps > 0
            else 0
        )


        # Approximately 1 saved frame per second.
        frame_interval = max(
            1,
            int(round(fps * SAMPLE_SECONDS))
        )


        print(
            f"\n{person_id}"
            f" | FPS={fps:.2f}"
            f" | frames={total_frames}"
            f" | duration={duration:.1f}s"
            f" | interval={frame_interval}"
        )


        frame_number = 0
        saved_for_person = 0


        # =================================================
        # READ VIDEO
        # =================================================

        while True:

            success, frame = cap.read()

            if not success:
                break


            if (
                frame_number % frame_interval == 0
                and
                saved_for_person < MAX_FRAMES_PER_PERSON
            ):

                file_name = (
                    f"{person_id}"
                    f"_frame_{frame_number:06d}.jpg"
                )

                output_file = (
                    person_output
                    / file_name
                )

                save_success = cv2.imwrite(
                    str(output_file),
                    frame
                )

                if save_success:

                    timestamp_seconds = (
                        frame_number / fps
                        if fps > 0
                        else None
                    )

                    metadata.append(
                        {
                            "person_id": person_id,
                            "label": label,
                            "video_file": video_path.name,
                            "frame_number": frame_number,
                            "timestamp_seconds": timestamp_seconds,
                            "fps": fps,
                            "video_total_frames": total_frames,
                            "image_path": str(output_file),
                        }
                    )

                    saved_for_person += 1
                    total_saved += 1


            frame_number += 1


            if (
                saved_for_person
                >= MAX_FRAMES_PER_PERSON
            ):
                break


        cap.release()

        print(
            "Saved:",
            saved_for_person
        )


# =========================================================
# SAVE METADATA
# =========================================================

df = pd.DataFrame(metadata)

df.to_csv(
    METADATA_PATH,
    index=False
)


# =========================================================
# SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("\nVideos processed:")
print(total_videos)

print("\nFrames saved:")
print(total_saved)

print("\nFailed videos:")
print(len(failed_videos))


if len(df) > 0:

    print("\nFrames by label:")
    print(
        df["label"].value_counts()
    )

    print("\nPeople by label:")
    print(
        df.groupby("label")[
            "person_id"
        ].nunique()
    )

    print("\nFrames per person:")
    print(
        df.groupby("person_id")
        .size()
        .describe()
    )


print("\nMetadata:")
print(METADATA_PATH)

print("\nImages:")
print(OUTPUT_PATH)

print("\n" + "=" * 70)
print("STEP 8B COMPLETE")
print("=" * 70)