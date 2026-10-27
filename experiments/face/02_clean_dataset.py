from pathlib import Path
from collections import Counter
import pandas as pd
import re


# =========================================================
# 1. PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = PROJECT_ROOT / "YFP_Dataset_Updated"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "face"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("STEP 2 - YFP DATA CLEANING")
print("=" * 70)

print("\nDataset:")
print(DATASET_PATH)


# =========================================================
# 2. IMAGE EXTENSIONS
# =========================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# =========================================================
# 3. NORMALIZE REGION
# =========================================================

def normalize_region(folder_name):

    name = folder_name.lower()

    if "eyebrow" in name:
        return "Eyebrow"

    elif "eye" in name:
        return "Eye"

    elif "mouth" in name:
        return "Mouth"

    return "Unknown"


# =========================================================
# 4. NORMALIZE SEVERITY LABEL
# =========================================================

def normalize_label(folder_name):

    name = folder_name.lower().strip()

    # IMPORTANT:
    # check moderate severe BEFORE moderate

    if "moderate severe" in name:
        return "Moderate Severe"

    elif "mild" in name:
        return "Mild"

    elif "moderate" in name:
        return "Moderate"

    elif "severe" in name:
        return "Severe"

    return "Unknown"


# =========================================================
# 5. DETECT AUGMENTATION TYPE
# =========================================================

def detect_augmentation(filename):

    name = filename.lower()

    if "rotated" in name:
        return "rotated"

    elif "translated" in name:
        return "translated"

    elif "noisy" in name:
        return "noisy"

    return "original"


# =========================================================
# 6. REMOVE AUGMENTATION PREFIX
# =========================================================

def remove_augmentation_prefix(filename):

    stem = Path(filename).stem

    clean = stem

    # rotated_
    clean = re.sub(
        r"^rotated_",
        "",
        clean,
        flags=re.IGNORECASE
    )

    # translated_
    clean = re.sub(
        r"^translated_",
        "",
        clean,
        flags=re.IGNORECASE
    )

    # noisy_25dB_
    # noisy_20dB_
    # etc.
    clean = re.sub(
        r"^noisy_[^_]+_",
        "",
        clean,
        flags=re.IGNORECASE
    )

    return clean


# =========================================================
# 7. TRY TO EXTRACT LEADING SUBJECT/SAMPLE NUMBER
# =========================================================

def extract_leading_id(clean_name):

    match = re.match(
        r"^(\d+)_",
        clean_name
    )

    if match:
        return match.group(1)

    return None


# =========================================================
# 8. READ ALL IMAGES
# =========================================================

records = []

for image_path in DATASET_PATH.rglob("*"):

    if (
        not image_path.is_file()
        or image_path.suffix.lower() not in IMAGE_EXTENSIONS
    ):
        continue

    relative_path = image_path.relative_to(DATASET_PATH)

    parts = relative_path.parts

    # Expected:
    # Region / Severity folder / filename

    region_folder = parts[0] if len(parts) >= 1 else ""

    severity_folder = (
        parts[1]
        if len(parts) >= 2
        else ""
    )

    region = normalize_region(region_folder)

    severity = normalize_label(severity_folder)

    augmentation = detect_augmentation(
        image_path.name
    )

    base_name = remove_augmentation_prefix(
        image_path.name
    )

    leading_id = extract_leading_id(
        base_name
    )

    records.append({

        "file_path":
            str(image_path),

        "relative_path":
            str(relative_path),

        "region":
            region,

        "severity":
            severity,

        "severity_folder":
            severity_folder,

        "file_name":
            image_path.name,

        "augmentation":
            augmentation,

        "base_name":
            base_name,

        "leading_id":
            leading_id
    })


df = pd.DataFrame(records)


# =========================================================
# 9. BASIC SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print("\nTotal images:")
print(len(df))


print("\nRegions:")
print(
    df["region"]
    .value_counts(dropna=False)
)


print("\nSeverity labels:")
print(
    df["severity"]
    .value_counts(dropna=False)
)


print("\nAugmentation types:")
print(
    df["augmentation"]
    .value_counts(dropna=False)
)


# =========================================================
# 10. REGION x SEVERITY
# =========================================================

print("\n" + "=" * 70)
print("REGION x SEVERITY")
print("=" * 70)

region_severity = pd.crosstab(
    df["region"],
    df["severity"]
)

print(region_severity)


# =========================================================
# 11. AUGMENTATION x REGION
# =========================================================

print("\n" + "=" * 70)
print("REGION x AUGMENTATION")
print("=" * 70)

region_augmentation = pd.crosstab(
    df["region"],
    df["augmentation"]
)

print(region_augmentation)


# =========================================================
# 12. CHECK UNKNOWN LABELS
# =========================================================

unknown_region = df[
    df["region"] == "Unknown"
]

unknown_severity = df[
    df["severity"] == "Unknown"
]

print("\n" + "=" * 70)
print("UNKNOWN CHECK")
print("=" * 70)

print(
    "Unknown regions:",
    len(unknown_region)
)

print(
    "Unknown severity labels:",
    len(unknown_severity)
)


if len(unknown_severity) > 0:

    print("\nUnknown severity folders:")

    print(
        unknown_severity[
            "severity_folder"
        ]
        .value_counts()
    )


# =========================================================
# 13. CHECK LEADING IDs
# =========================================================

print("\n" + "=" * 70)
print("LEADING ID CHECK")
print("=" * 70)

print(
    "Images with leading ID:",
    df["leading_id"].notna().sum()
)

print(
    "Images without leading ID:",
    df["leading_id"].isna().sum()
)


print("\nUnique leading IDs:")

ids = sorted(
    df["leading_id"]
    .dropna()
    .unique(),
    key=lambda x: int(x)
)

print(ids)

print(
    "\nNumber of unique IDs:",
    len(ids)
)


# =========================================================
# 14. SHOW EXAMPLES
# =========================================================

print("\n" + "=" * 70)
print("ORIGINAL EXAMPLES")
print("=" * 70)

original_examples = df[
    df["augmentation"] == "original"
][
    [
        "region",
        "severity",
        "file_name",
        "base_name",
        "leading_id"
    ]
].head(20)

print(
    original_examples.to_string(
        index=False
    )
)


print("\n" + "=" * 70)
print("AUGMENTED EXAMPLES")
print("=" * 70)

augmented_examples = df[
    df["augmentation"] != "original"
][
    [
        "region",
        "severity",
        "file_name",
        "augmentation",
        "base_name",
        "leading_id"
    ]
].head(30)

print(
    augmented_examples.to_string(
        index=False
    )
)


# =========================================================
# 15. BASE NAME DUPLICATION CHECK
# =========================================================

print("\n" + "=" * 70)
print("BASE NAME CHECK")
print("=" * 70)

base_counts = (
    df
    .groupby(
        [
            "region",
            "severity",
            "base_name"
        ]
    )
    .size()
    .reset_index(
        name="count"
    )
)

possible_related = base_counts[
    base_counts["count"] > 1
]

print(
    "Repeated base-name groups:",
    len(possible_related)
)

if len(possible_related) > 0:

    print("\nExamples:")

    print(
        possible_related
        .head(30)
        .to_string(index=False)
    )


# =========================================================
# 16. SAVE CLEAN MANIFEST
# =========================================================

clean_manifest = (
    OUTPUT_PATH /
    "face_clean_manifest.csv"
)

df.to_csv(
    clean_manifest,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 70)
print("SAVED")
print("=" * 70)

print(
    "\nClean manifest:"
)

print(
    clean_manifest
)


# =========================================================
# 17. FINAL MESSAGE
# =========================================================

print("\n" + "=" * 70)
print("STEP 2 COMPLETE")
print("=" * 70)

print(
    "\nDO NOT TRAIN THE MODEL YET."
)

print(
    "Inspect subject/sample IDs before train-test splitting."
)