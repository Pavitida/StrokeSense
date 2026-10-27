from pathlib import Path
import pandas as pd
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "face"
    / "face_hog_metadata.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "face"
    / "face_grouped_metadata.csv"
)

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("STEP 4A - BUILD SOURCE IMAGE GROUPS")
print("=" * 70)

print("\nTotal images:", len(df))


# =========================================================
# NORMALIZE FILENAME
# =========================================================

def normalize_source_name(filename):

    name = Path(filename).stem.lower()

    previous = None

    while previous != name:

        previous = name

        # crop
        name = re.sub(
            r"^cropped_",
            "",
            name
        )

        # noise
        # noisy_20db_
        # noisy_25db_
        name = re.sub(
            r"^noisy_[^_]+_",
            "",
            name
        )

        # translation
        name = re.sub(
            r"^translated_",
            "",
            name
        )

        # rotation
        name = re.sub(
            r"^rotated_",
            "",
            name
        )

        # negative rotation parameter
        name = re.sub(
            r"^minus\d+_",
            "",
            name
        )

    return name


df["source_name"] = (
    df["file_name"]
    .apply(normalize_source_name)
)


# =========================================================
# CREATE SOURCE FAMILY ID
#
# IMPORTANT:
# Region + Severity are included intentionally.
# We are grouping augmentation relatives,
# NOT claiming this is a patient ID.
# =========================================================

df["source_group"] = (
    df["region"].astype(str)
    + "__"
    + df["severity"].astype(str)
    + "__"
    + df["source_name"].astype(str)
)


# =========================================================
# SUMMARY
# =========================================================

print(
    "\nUnique source groups:",
    df["source_group"].nunique()
)

sizes = (
    df.groupby("source_group")
    .size()
)

print(
    "Groups containing >1 image:",
    (sizes > 1).sum()
)

print(
    "Largest group size:",
    sizes.max()
)


print("\nGroup size distribution:")

print(
    sizes.value_counts()
    .sort_index()
    .head(20)
)


# =========================================================
# LABEL CONSISTENCY
# =========================================================

label_check = (
    df.groupby("source_group")["severity"]
    .nunique()
)

bad = label_check[
    label_check > 1
]

print(
    "\nGroups with multiple severity labels:",
    len(bad)
)


# =========================================================
# EXAMPLES
# =========================================================

print("\n" + "=" * 70)
print("GROUP EXAMPLES")
print("=" * 70)

repeated_groups = (
    sizes[sizes > 1]
    .index[:10]
)

for group in repeated_groups:

    subset = df[
        df["source_group"] == group
    ]

    print("\nGROUP:", group)

    for filename in subset["file_name"]:
        print("  ", filename)


# =========================================================
# SAVE
# =========================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 70)
print("SAVED")
print("=" * 70)

print(OUTPUT_FILE)

print("\nSTEP 4A COMPLETE")