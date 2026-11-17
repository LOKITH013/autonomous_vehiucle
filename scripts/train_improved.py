"""
M5 - Improved YOLOv8s Training on Full IDD-D Dataset (All Recommendations Applied)

Improvements over M4 (YOLOv8n baseline):
  1. Model upgrade:       YOLOv8n (3.0M params) -> YOLOv8s (11.2M params)
  2. Full dataset:        all 41,962 images (train=33569, val=4196, test=4197)
  3. Input resolution:    640px -> 960px  (improves detection of small / distant objects)
  4. Training duration:   50 epochs + cosine LR scheduling
  5. Optimizer:           AdamW with warmup + weight decay
  6. Augmentation:        mosaic, mixup, copy-paste enabled

Usage:
    python scripts/train_improved.py
    python scripts/train_improved.py --epochs 100 --batch 4 --imgsz 1280

Output:
    Best checkpoint -> models/detection/best.pt   (auto-replaced on improvement)
    Training runs   -> runs/detect/astra_m5_yolov8s/
"""

import argparse
import os
import shutil
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT_DIR)


def parse_args():
    p = argparse.ArgumentParser(description="Astra ADAS M5 - Improved YOLOv8s Training")
    p.add_argument("--model",   default="yolov8s.pt",
                   help="Backbone to fine-tune: yolov8s.pt | yolov8m.pt | yolov11s.pt")
    p.add_argument("--epochs",  type=int, default=50,
                   help="Training epochs (50 recommended; 100 for best results)")
    p.add_argument("--imgsz",   type=int, default=960,
                   help="Input image resolution in px (960 or 1280 for best small-object detection)")
    p.add_argument("--batch",   type=int, default=8,
                   help="Batch size. Reduce to 4 if GPU runs out of memory")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device",  default="",
                   help="Device: empty=auto, '0'=GPU 0, 'cpu'=CPU only")
    p.add_argument("--data",    default="IDDDetectionsYOLODataset/data.yaml")
    p.add_argument("--project", default="runs/detect")
    p.add_argument("--name",    default="astra_m5_yolov8s")
    return p.parse_args()


def main():
    args = parse_args()

    # -- Validate dataset yaml exists --
    data_yaml = os.path.join(ROOT_DIR, args.data)
    if not os.path.isfile(data_yaml):
        print("[ERROR] Dataset config not found: " + data_yaml)
        sys.exit(1)

    # -- Report dataset split sizes --
    dataset_root = os.path.join(ROOT_DIR, "IDDDetectionsYOLODataset")
    total = 0
    print("=" * 62)
    print("   ASTRA ADAS - M5 IMPROVED TRAINING (Full IDD-D Dataset)")
    print("=" * 62)
    print("   Backbone   : " + args.model)
    print("   Epochs     : " + str(args.epochs))
    print("   Resolution : " + str(args.imgsz) + "px")
    print("   Batch size : " + str(args.batch))
    print("   Optimizer  : AdamW + cosine LR schedule")
    for split in ("train", "val", "test"):
        img_dir = os.path.join(dataset_root, split, "images")
        if os.path.isdir(img_dir):
            n = len([f for f in os.listdir(img_dir)
                     if f.lower().endswith((".jpg", ".png"))])
            print("   " + split.upper().ljust(8) + ": " + str(n).rjust(6) + " images")
            total += n
        else:
            print("   " + split.upper().ljust(8) + ": MISSING - " + img_dir)
    print("   " + "TOTAL".ljust(8) + ": " + str(total).rjust(6) + " images")
    print("=" * 62)

    # -- Import YOLO --
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    model = YOLO(args.model)

    # -- Train --
    print("[INFO] Starting training: " + os.path.join(args.project, args.name))
    results = model.train(
        # Core
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device if args.device else None,

        # Output
        project=os.path.join(ROOT_DIR, args.project),
        name=args.name,
        save=True,
        save_period=10,          # checkpoint every 10 epochs

        # Optimizer & LR (cosine annealing for better convergence)
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3,
        weight_decay=0.0005,
        momentum=0.937,

        # Augmentation (IDD-safe settings)
        mosaic=1.0,              # critical for small-object detection
        mixup=0.15,
        copy_paste=0.1,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0001,
        flipud=0.0,              # no vertical flip (driving scenes)
        fliplr=0.5,

        # NMS settings
        iou=0.55,                # matches updated config.yaml
        conf=0.001,              # low threshold during training NMS

        # Logging
        plots=True,
        val=True,
        verbose=True,
    )

    # -- Copy best weights to models/detection/best.pt --
    run_dir = os.path.join(ROOT_DIR, args.project, args.name)
    best_src = os.path.join(run_dir, "weights", "best.pt")
    best_dst = os.path.join(ROOT_DIR, "models", "detection", "best.pt")
    os.makedirs(os.path.dirname(best_dst), exist_ok=True)

    if os.path.isfile(best_src):
        shutil.copy2(best_src, best_dst)
        print("[SUCCESS] Best weights saved to: " + best_dst)
        print("[INFO]    Full training artefacts: " + run_dir)
    else:
        print("[WARNING] best.pt not found at: " + best_src)

    # -- Print final metrics --
    if results and hasattr(results, "results_dict"):
        rd = results.results_dict
        print("=" * 62)
        print("   TRAINING COMPLETE - Metrics on val split:")
        print("=" * 62)
        for key in ["metrics/mAP50(B)", "metrics/mAP50-95(B)",
                    "metrics/precision(B)", "metrics/recall(B)"]:
            if key in rd:
                label = key.replace("metrics/", "").replace("(B)", "")
                print("   " + label.ljust(20) + ": " + str(round(rd[key], 4)))
        print("=" * 62)
        print("")
        print("Next step - run full benchmark vs old model:")
        print("  python -m src.evaluation.benchmark_m4 --num-images all --split val")

    return results


if __name__ == "__main__":
    main()
