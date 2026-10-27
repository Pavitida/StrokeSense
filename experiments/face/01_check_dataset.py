from pathlib import Path
from collections import Counter
from PIL import Image
import pandas as pd


# =========================================================
# 1. PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "YFP_Dataset_Updated"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "face"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("YFP DATASET CHECK")
print("=" * 60)

print("\nProject root:")
print(PROJECT_ROOT)

print("\nDataset path:")
print(DATASET_PATH)

print("\nDataset exists:", DATASET_PATH.exists())


# =========================================================
# 2. STOP IF DATASET DOES NOT EXIST
# =========================================================

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"\nไม่พบ Dataset ที่:\n{DATASET_PATH}\n"
        "ตรวจว่า YFP_Dataset_Updated อยู่ใน STROKE_PROJECT"
    )


# =========================================================
# 3. CHECK MAIN FOLDERS
# =========================================================

print("\n" + "=" * 60)
print("MAIN FOLDERS")
print("=" * 60)

main_folders = [
    item.name
    for item in DATASET_PATH.iterdir()
    if item.is_dir()
]

for folder in sorted(main_folders):
    print("•", folder)


# =========================================================
# 4. FIND ALL IMAGES
# =========================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

image_paths = [
    path
    for path in DATASET_PATH.rglob("*")
    if path.is_file()
    and path.suffix.lower() in IMAGE_EXTENSIONS
]

print("\n" + "=" * 60)
print("IMAGE COUNT")
print("=" * 60)

print("Total images:", len(image_paths))


# =========================================================
# 5. COUNT IMAGES BY TOP-LEVEL FOLDER
# =========================================================

region_counter = Counter()

for image_path in image_paths:

    relative_path = image_path.relative_to(DATASET_PATH)
    parts = relative_path.parts

    if len(parts) > 0:
        region = parts[0]
    else:
        region = "Unknown"

    region_counter[region] += 1


print("\nImages by folder:")

for region, count in sorted(region_counter.items()):
    print(f"{region}: {count}")


# =========================================================
# 6. BUILD DATASET MANIFEST
# =========================================================

records = []

for image_path in image_paths:

    relative_path = image_path.relative_to(DATASET_PATH)
    parts = relative_path.parts

    region = parts[0] if len(parts) >= 1 else "Unknown"

    # เก็บโครงสร้างทั้งหมดไว้ก่อน
    parent_folders = list(parts[:-1])

    records.append({
        "file_path": str(image_path),
        "relative_path": str(relative_path),
        "region": region,
        "parent_folders": "/".join(parent_folders),
        "file_name": image_path.name,
        "extension": image_path.suffix.lower()
    })


df = pd.DataFrame(records)

print("\n" + "=" * 60)
print("DATASET PREVIEW")
print("=" * 60)

if len(df) > 0:
    print(df.head(10).to_string(index=False))
else:
    print("ไม่พบรูปภาพ")


# =========================================================
# 7. CHECK CORRUPTED IMAGES
# =========================================================

print("\n" + "=" * 60)
print("CHECKING CORRUPTED IMAGES")
print("=" * 60)

bad_files = []

for index, image_path in enumerate(image_paths, start=1):

    try:

        with Image.open(image_path) as image:
            image.verify()

    except Exception as error:

        bad_files.append({
            "file_path": str(image_path),
            "error": str(error)
        })

    if index % 500 == 0:
        print(
            f"Checked {index}/{len(image_paths)} images..."
        )


print("\nCorrupted images:", len(bad_files))


# =========================================================
# 8. SAVE RESULTS
# =========================================================

manifest_file = OUTPUT_PATH / "face_manifest.csv"

df.to_csv(
    manifest_file,
    index=False,
    encoding="utf-8-sig"
)

print("\nManifest saved:")
print(manifest_file)


if bad_files:

    bad_df = pd.DataFrame(bad_files)

    bad_file_path = (
        OUTPUT_PATH / "corrupted_images.csv"
    )

    bad_df.to_csv(
        bad_file_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nCorrupted image list saved:")
    print(bad_file_path)


# =========================================================
# 9. FINAL SUMMARY
# =========================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print("Dataset exists: YES")
print("Total images:", len(image_paths))
print("Corrupted images:", len(bad_files))

print("\nRegions found:")

for region, count in sorted(region_counter.items()):
    print(f"  {region}: {count}")

print("\nSTEP 1 CHECK COMPLETE")
print("=" * 60)