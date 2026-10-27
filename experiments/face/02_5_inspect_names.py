from pathlib import Path
import pandas as pd
import re

# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "outputs" / "face" / "face_clean_manifest.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "face"

df = pd.read_csv(MANIFEST_PATH)

print("=" * 80)
print("STEP 2.5 - INSPECT YFP FILE NAMING")
print("=" * 80)

print("\nTotal images:", len(df))


# =========================================================
# FUNCTION: REMOVE AUGMENTATION PREFIXES
# =========================================================

def strip_augmentation(name):

    stem = Path(name).stem

    # Keep removing prefixes until no rule matches
    previous = None

    while previous != stem:

        previous = stem

        # noisy_25dB_
        # noisy_20dB_
        stem = re.sub(
            r"^noisy_[^_]+_",
            "",
            stem,
            flags=re.IGNORECASE
        )

        # translated_
        stem = re.sub(
            r"^translated_",
            "",
            stem,
            flags=re.IGNORECASE
        )

        # rotated_
        stem = re.sub(
            r"^rotated_",
            "",
            stem,
            flags=re.IGNORECASE
        )

    return stem


df["stripped_name"] = df["file_name"].apply(strip_augmentation)


# =========================================================
# CLASSIFY NAME PATTERNS
# =========================================================

def classify_pattern(name):

    if re.match(r"^\d+_", name):
        return "number_prefix"

    if re.match(r"^minus\d+_", name, re.IGNORECASE):
        return "minus_prefix"

    return "other"


df["name_pattern"] = df["stripped_name"].apply(classify_pattern)


print("\n" + "=" * 80)
print("NAME PATTERNS")
print("=" * 80)

print(df["name_pattern"].value_counts(dropna=False))


# =========================================================
# PRINT UNIQUE FIRST TOKENS
# =========================================================

df["first_token"] = (
    df["stripped_name"]
    .astype(str)
    .str.split("_")
    .str[0]
)

print("\n" + "=" * 80)
print("FIRST TOKENS")
print("=" * 80)

token_counts = (
    df["first_token"]
    .value_counts()
    .reset_index()
)

token_counts.columns = [
    "first_token",
    "count"
]

print(token_counts.head(100).to_string(index=False))


# =========================================================
# INSPECT EACH REGION + SEVERITY
# =========================================================

print("\n" + "=" * 80)
print("SAMPLES BY REGION / SEVERITY")
print("=" * 80)

for region in sorted(df["region"].unique()):

    for severity in sorted(df["severity"].unique()):

        subset = df[
            (df["region"] == region) &
            (df["severity"] == severity)
        ]

        if len(subset) == 0:
            continue

        print("\n" + "-" * 80)
        print(region, "|", severity)
        print("Images:", len(subset))

        print("\nExample ORIGINAL filenames:")

        originals = subset[
            subset["augmentation"] == "original"
        ]["file_name"].head(15)

        for name in originals:
            print(" ", name)

        print("\nExample AUGMENTED filenames:")

        augmented = subset[
            subset["augmentation"] != "original"
        ]["file_name"].head(15)

        for name in augmented:
            print(" ", name)


# =========================================================
# FIND NAMES CONTAINING minus
# =========================================================

print("\n" + "=" * 80)
print("MINUS EXAMPLES")
print("=" * 80)

minus_rows = df[
    df["stripped_name"]
    .str.contains(
        "minus",
        case=False,
        na=False
    )
]

print("Total minus names:", len(minus_rows))

print(
    minus_rows[
        [
            "region",
            "severity",
            "file_name",
            "stripped_name"
        ]
    ]
    .head(50)
    .to_string(index=False)
)


# =========================================================
# EXTRACT CONTENT AROUND REGION WORD
# =========================================================

def remove_region_counter(name):

    # Examples:
    # 18_M.S_eyebrow188
    # 24_Mi_eye10
    # something_mouth123

    cleaned = re.sub(
        r"(eye|eyebrow|mouth)\d+$",
        r"\1",
        name,
        flags=re.IGNORECASE
    )

    return cleaned


df["without_frame_counter"] = (
    df["stripped_name"]
    .apply(remove_region_counter)
)


print("\n" + "=" * 80)
print("WITHOUT FRAME COUNTER EXAMPLES")
print("=" * 80)

print(
    df[
        [
            "region",
            "severity",
            "file_name",
            "stripped_name",
            "without_frame_counter"
        ]
    ]
    .head(50)
    .to_string(index=False)
)


# =========================================================
# GROUP COUNTS AFTER NORMALIZATION
# =========================================================

group_counts = (
    df
    .groupby(
        [
            "region",
            "severity",
            "without_frame_counter"
        ]
    )
    .size()
    .reset_index(name="images")
    .sort_values(
        "images",
        ascending=False
    )
)


print("\n" + "=" * 80)
print("POSSIBLE GROUPS")
print("=" * 80)

print(
    group_counts
    .head(100)
    .to_string(index=False)
)


# =========================================================
# SAVE INSPECTION TABLE
# =========================================================

inspection_path = (
    OUTPUT_PATH /
    "filename_inspection.csv"
)

df.to_csv(
    inspection_path,
    index=False,
    encoding="utf-8-sig"
)

groups_path = (
    OUTPUT_PATH /
    "possible_groups.csv"
)

group_counts.to_csv(
    groups_path,
    index=False,
    encoding="utf-8-sig"
)


print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print(inspection_path)
print(groups_path)

print("\nSTEP 2.5 COMPLETE")