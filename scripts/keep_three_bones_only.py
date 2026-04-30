import os
import shutil
import csv

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

KEEP_BONES = [
    "radius",
    "ulna",
    "humerus"
]

IGNORE_DIRS = {
    ".git", "venv", "__pycache__", "outputs", "models",
    "scripts", ".ipynb_checkpoints"
}

QUARANTINE_DIR = "outputs/removed_other_bones_quarantine"
REPORT_PATH = "outputs/removed_other_bones_report.csv"

removed = []
kept = []

def path_has_keep_bone(path):
    p = path.lower().replace("\\", "/")
    return any(bone in p for bone in KEEP_BONES)

def is_dataset_image(path):
    p = path.lower().replace("\\", "/")

    if not path.lower().endswith(IMAGE_EXTS):
        return False

    # عشان مايلمسش assets أو reports أو أي صور مش dataset
    dataset_keywords = [
        "dataset",
        "data",
        "train",
        "test",
        "val",
        "valid",
        "fractured",
        "not_fractured",
        "normal"
    ]

    return any(word in p for word in dataset_keywords)

for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

    for file in files:
        path = os.path.join(root, file)

        if not is_dataset_image(path):
            continue

        if path_has_keep_bone(path):
            kept.append(path)
            continue

        safe_name = path.replace("./", "").replace("/", "__")
        target = os.path.join(QUARANTINE_DIR, safe_name)
        os.makedirs(os.path.dirname(target), exist_ok=True)

        shutil.move(path, target)

        removed.append({
            "original_path": path,
            "moved_to": target,
            "reason": "not radius, ulna, or humerus"
        })

with open(REPORT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["original_path", "moved_to", "reason"])
    writer.writeheader()
    writer.writerows(removed)

print("Cleaning finished.")
print("Kept images:", len(kept))
print("Removed other-bone images:", len(removed))
print("Report:", REPORT_PATH)
print("Quarantine:", QUARANTINE_DIR)
