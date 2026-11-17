"""
M2 — Dataset integrity inspection (Sec 18, Sec 35).

This script does NOT assume any particular IDD annotation schema — it
inspects whatever is actually present under data/idd/raw/ and reports facts,
because "we must not invent the IDD annotation structure" (Sec 35).

Usage:
    python scripts/inspect_idd.py --root data/idd/raw
"""
import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
ANNOT_EXTS = {".json", ".xml", ".txt"}


def file_hash(path, block_size=65536):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()


def walk_files(root):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            yield os.path.join(dirpath, fn)


def main(root):
    if not os.path.isdir(root):
        print(f"[FAIL] Root path does not exist: {root}")
        print("Download the IDD dataset from the official source first "
              "(Sec 17) and place it under this path. This script does not "
              "download or redistribute the dataset.")
        return

    images, annots, others = [], [], []
    for path in walk_files(root):
        ext = os.path.splitext(path)[1].lower()
        if ext in IMAGE_EXTS:
            images.append(path)
        elif ext in ANNOT_EXTS:
            annots.append(path)
        else:
            others.append(path)

    print("=" * 60)
    print("IDD RAW DATA INVENTORY")
    print("=" * 60)
    print(f"Root:                {root}")
    print(f"Image files found:   {len(images)}")
    print(f"Annotation-like files found: {len(annots)}  (json/xml/txt)")
    print(f"Other files found:   {len(others)}")

    if not images:
        print("\n[WARN] No image files found. Verify extraction path.")
        return

    # --- pairing check: does every image have a same-stem annotation? ---
    img_stems = {os.path.splitext(os.path.basename(p))[0]: p for p in images}
    ann_stems = {os.path.splitext(os.path.basename(p))[0]: p for p in annots}
    missing_annot = [s for s in img_stems if s not in ann_stems]
    orphan_annot = [s for s in ann_stems if s not in img_stems]

    print(f"\nImages missing a same-name annotation file: {len(missing_annot)}")
    if missing_annot[:5]:
        print("  e.g.:", missing_annot[:5])
    print(f"Annotation files with no matching image:     {len(orphan_annot)}")
    if orphan_annot[:5]:
        print("  e.g.:", orphan_annot[:5])

    # --- duplicate check (content hash on a sample, hashing all can be slow) ---
    print("\nChecking for duplicate images (hashing up to 2000 files)...")
    hashes = defaultdict(list)
    for p in images[:2000]:
        try:
            hashes[file_hash(p)].append(p)
        except OSError:
            print(f"  [CORRUPT/UNREADABLE] {p}")
    dupes = {h: ps for h, ps in hashes.items() if len(ps) > 1}
    print(f"Duplicate groups found (sampled): {len(dupes)}")

    # --- corrupted image check (basic header check via OpenCV) ---
    try:
        import cv2
        corrupt = []
        for p in images[:2000]:
            img = cv2.imread(p)
            if img is None:
                corrupt.append(p)
        print(f"Unreadable/corrupted images (sampled): {len(corrupt)}")
        if corrupt[:5]:
            print("  e.g.:", corrupt[:5])
    except ImportError:
        print("[SKIP] opencv-python not installed; skipping corruption check.")

    # --- inspect one annotation file to reveal actual schema ---
    if annots:
        sample = annots[0]
        print(f"\nSample annotation file: {sample}")
        ext = os.path.splitext(sample)[1].lower()
        if ext == ".json":
            with open(sample, "r") as f:
                try:
                    data = json.load(f)
                    print("Top-level JSON keys:", list(data.keys()) if isinstance(data, dict) else type(data))
                except json.JSONDecodeError:
                    print("[WARN] Could not parse as JSON.")
        else:
            with open(sample, "r", errors="ignore") as f:
                print("First 500 chars:\n", f.read(500))

    # --- directory-level split hints (video/sequence folders vs flat) ---
    top_level_dirs = set()
    for p in images:
        rel = os.path.relpath(p, root)
        parts = rel.split(os.sep)
        if len(parts) > 1:
            top_level_dirs.add(parts[0])
    print(f"\nTop-level subfolders under images (potential split/sequence "
          f"boundaries, see Sec 18 leakage warning): {sorted(top_level_dirs)[:20]}")

    print("\nNEXT STEP: open the sample annotation file above manually, "
          "confirm the schema against the official IDD documentation, THEN "
          "write scripts/convert_idd_to_yolo.py against the CONFIRMED schema. "
          "Do not assume a schema from this report alone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/idd/raw")
    args = parser.parse_args()
    main(args.root)
