import os
import cv2
import csv
import hashlib
import shutil
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

IGNORE_DIRS = {
    ".git", "venv", "__pycache__", "outputs", "models",
    "scripts", ".ipynb_checkpoints", "bad_images_quarantine"
}

BAD_DIR = "outputs/bad_images_quarantine"
REPORT = "outputs/bad_images_report.csv"

MIN_WIDTH = 80
MIN_HEIGHT = 80
MIN_FILE_SIZE = 1024
MIN_STD_INTENSITY = 3.0

seen_hashes = {}
bad_rows = []

def file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def mark_bad(path, reason):
    safe_name = path.replace("./", "").replace("/", "__")
    target = os.path.join(BAD_DIR, safe_name)

    os.makedirs(os.path.dirname(target), exist_ok=True)

    if os.path.exists(path):
        shutil.move(path, target)

    bad_rows.append({
        "original_path": path,
        "moved_to": target,
        "reason": reason
    })

def is_image_corrupt_pil(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return False
    except Exception:
        return True

total = 0

for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

    for file in files:
        if not file.lower().endswith(IMAGE_EXTS):
            continue

        path = os.path.join(root, file)
        total += 1

        try:
            size = os.path.getsize(path)
            if size < MIN_FILE_SIZE:
                mark_bad(path, "very small file size")
                continue

            if is_image_corrupt_pil(path):
                mark_bad(path, "corrupt or incomplete image")
                continue

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                mark_bad(path, "opencv cannot read image")
                continue

            h, w = img.shape[:2]

            if w < MIN_WIDTH or h < MIN_HEIGHT:
                mark_bad(path, "image resolution too small")
                continue

            std_value = float(np.std(img))
            mean_value = float(np.mean(img))

            if std_value < MIN_STD_INTENSITY:
                mark_bad(path, "almost blank image")
                continue

            if mean_value < 5:
                mark_bad(path, "too dark image")
                continue

            if mean_value > 250:
                mark_bad(path, "too bright image")
                continue

            md5 = file_hash(path)

            if md5 in seen_hashes:
                mark_bad(path, f"duplicate image of {seen_hashes[md5]}")
                continue

            seen_hashes[md5] = path

        except Exception as e:
            mark_bad(path, f"error while checking image: {e}")

with open(REPORT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["original_path", "moved_to", "reason"])
    writer.writeheader()
    writer.writerows(bad_rows)

print("Scan finished.")
print("Total checked images:", total)
print("Bad images moved:", len(bad_rows))
print("Report saved to:", REPORT)
print("Bad images moved to:", BAD_DIR)
