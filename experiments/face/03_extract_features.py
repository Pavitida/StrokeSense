from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from skimage.feature import hog


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "face"
    / "face_clean_manifest.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "face"
)

OUTPUT_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# SETTINGS
# =========================================================

IMAGE_SIZE = (128, 128)

print("=" * 70)
print("STEP 3 - HOG FEATURE EXTRACTION")
print("=" * 70)

df = pd.read_csv(MANIFEST_PATH)

print("\nTotal images:", len(df))


# =========================================================
# SUBJECT / GROUP ID
# =========================================================

def extract_group_id(filename):

    name = Path(filename).stem.lower()

    # ---------------------------------------------
    # Remove augmentation prefixes repeatedly
    # ---------------------------------------------

    prefixes = [
        "cropped_",
        "rotated_",
        "translated_"
    ]

    changed = True

    while changed:

        changed = False

        for prefix in prefixes:

            if name.startswith(prefix):
                name = name[len(prefix):]
                changed = True

    # ---------------------------------------------
    # Remove noise prefix
    # noisy_25db_
    # noisy_20db_
    # ---------------------------------------------

    import re

    name = re.sub(
        r"^noisy_[^_]+_",
        "",
        name
    )

    # ---------------------------------------------
    # Handle rotation parameter
    # minus3_...
    # minus5_...
    # ---------------------------------------------

    name = re.sub(
        r"^minus\d+_",
        "",
        name
    )

    # ---------------------------------------------
    # First numeric token
    # ---------------------------------------------

    match = re.match(
        r"^(\d+)_",
        name
    )

    if match:
        return match.group(1)

    return None


df["group_id"] = (
    df["file_name"]
    .apply(extract_group_id)
)


# =========================================================
# CHECK GROUP IDS
# =========================================================

print("\nImages with group ID:")
print(df["group_id"].notna().sum())

print("\nImages without group ID:")
print(df["group_id"].isna().sum())

print("\nDetected group IDs:")

group_ids = sorted(
    df["group_id"]
    .dropna()
    .astype(str)
    .unique(),
    key=lambda x: int(x)
)

print(group_ids)

print(
    "\nNumber of groups:",
    len(group_ids)
)


# =========================================================
# HOG EXTRACTION
# =========================================================

def extract_hog(image_path):

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        return None

    image = cv2.resize(
        image,
        IMAGE_SIZE
    )

    # Normalize pixel values
    image = image.astype(
        np.float32
    ) / 255.0

    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True
    )

    return features


# =========================================================
# PROCESS DATASET
# =========================================================

features = []
metadata = []

failed = []

total = len(df)

print("\nExtracting HOG features...\n")


for index, row in df.iterrows():

    feature = extract_hog(
        row["file_path"]
    )

    if feature is None:

        failed.append(
            row["file_path"]
        )

        continue

    features.append(
        feature
    )

    metadata.append({

        "file_path":
            row["file_path"],

        "file_name":
            row["file_name"],

        "region":
            row["region"],

        "severity":
            row["severity"],

        "augmentation":
            row["augmentation"],

        "group_id":
            row["group_id"]
    })

    processed = index + 1

    if processed % 500 == 0:

        print(
            f"Processed "
            f"{processed}/{total}"
        )


# =========================================================
# CONVERT TO NUMPY
# =========================================================

X = np.array(
    features,
    dtype=np.float32
)

metadata_df = pd.DataFrame(
    metadata
)


print("\n" + "=" * 70)
print("FEATURE RESULT")
print("=" * 70)

print(
    "\nFeature matrix shape:",
    X.shape
)

print(
    "Metadata shape:",
    metadata_df.shape
)

print(
    "Failed images:",
    len(failed)
)


# =========================================================
# GROUP CHECK AFTER EXTRACTION
# =========================================================

print("\nGroup ID missing:")

print(
    metadata_df[
        "group_id"
    ].isna().sum()
)


print("\nImages per region:")

print(
    metadata_df[
        "region"
    ].value_counts()
)


print("\nImages per severity:")

print(
    metadata_df[
        "severity"
    ].value_counts()
)


# =========================================================
# SAVE FEATURES
# =========================================================

FEATURE_FILE = (
    OUTPUT_PATH
    / "face_hog_features.npy"
)

METADATA_FILE = (
    OUTPUT_PATH
    / "face_hog_metadata.csv"
)

np.save(
    FEATURE_FILE,
    X
)

metadata_df.to_csv(
    METADATA_FILE,
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# SAVE FAILED
# =========================================================

if failed:

    failed_df = pd.DataFrame({
        "file_path": failed
    })

    failed_df.to_csv(
        OUTPUT_PATH
        / "hog_failed_images.csv",
        index=False
    )


# =========================================================
# FINAL
# =========================================================

print("\n" + "=" * 70)
print("SAVED")
print("=" * 70)

print("\nFeatures:")
print(FEATURE_FILE)

print("\nMetadata:")
print(METADATA_FILE)

print("\nSTEP 3 COMPLETE")