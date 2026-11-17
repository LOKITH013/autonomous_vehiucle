"""
M4 — Formal Quantitative Evaluation & Benchmark.
Compares the COCO-pretrained baseline (yolov8n.pt) against the IDD fine-tuned model
(models/detection/best.pt) on an IDD-D validation or test split.

Evaluates:
  - AP@50 per class and mAP@50 across classes present in the selected split.
  - Precision, Recall, and F1 at a fixed confidence threshold.
  - Specific subset mAP for Indian-specific classes (autorickshaw, rider, animal, etc.).
  - Inference latency and FPS.
  - Generates comparison charts and a markdown benchmark report.

Usage:
  python -m src.evaluation.benchmark_m4 --num-images 200 --split val
  python -m src.evaluation.benchmark_m4 --num-images all
"""

import argparse
import glob
import os
import sys
import time
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO

# 15 IDD Classes in exact order from data.yaml
IDD_CLASSES = [
    "animal",           # 0
    "autorickshaw",     # 1
    "bicycle",          # 2
    "bus",              # 3
    "car",              # 4
    "caravan",          # 5
    "motorcycle",       # 6
    "person",           # 7
    "rider",            # 8
    "traffic light",    # 9
    "traffic sign",     # 10
    "trailer",          # 11
    "train",            # 12
    "truck",            # 13
    "vehicle fallback"  # 14
]

# Indian-specific classes absent or uniquely defined in Indian driving context
INDIAN_SPECIFIC_CLASSES = {
    "autorickshaw", "rider", "animal", "caravan", "trailer", "vehicle fallback"
}

# COCO 80 classes to IDD 15 classes mapping
COCO_TO_IDD = {
    0: 7,   # person -> person
    1: 2,   # bicycle -> bicycle
    2: 4,   # car -> car
    3: 6,   # motorcycle -> motorcycle
    5: 3,   # bus -> bus
    6: 12,  # train -> train
    7: 13,  # truck -> truck
    9: 9,   # traffic light -> traffic light
    11: 10, # stop sign -> traffic sign
    # Animals in COCO -> IDD animal (0)
    14: 0,  # bird
    15: 0,  # cat
    16: 0,  # dog
    17: 0,  # horse
    18: 0,  # sheep
    19: 0,  # cow
    20: 0,  # elephant
    21: 0,  # bear
    22: 0,  # zebra
    23: 0,  # giraffe
}

AP_CONF_THRESHOLD = 0.001
OPERATING_CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5


def compute_iou(box1, box2):
    """
    Computes IoU between box1 [x1, y1, x2, y2] and box2 [x1, y1, x2, y2].
    """
    xx1 = max(box1[0], box2[0])
    yy1 = max(box1[1], box2[1])
    xx2 = min(box1[2], box2[2])
    yy2 = min(box1[3], box2[3])

    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    intersection = w * h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def calculate_ap(recalls, precisions):
    """
    Computes COCO-style 101-point interpolated Average Precision.
    """
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    recall_grid = np.linspace(0.0, 1.0, 101)
    interpolated_precision = np.interp(recall_grid, mrec, mpre)
    return float(
        np.sum(
            (interpolated_precision[:-1] + interpolated_precision[1:])
            * 0.5
            * np.diff(recall_grid)
        )
    )


def evaluate_model(
    model,
    image_paths,
    label_dir,
    is_coco=False,
    conf_thresh=AP_CONF_THRESHOLD,
    iou_thresh=IOU_THRESHOLD,
    imgsz=640,
    operating_conf_thresholds=None,
    operating_default_threshold=OPERATING_CONF_THRESHOLD,
):
    """
    Evaluates a model over a set of images against ground truth YOLO labels.
    Returns per-class metrics and timing info.
    """
    # GT count per class
    gt_counts = defaultdict(int)
    # class_id -> list of {'conf': float, 'is_tp': bool}
    class_preds = defaultdict(list)

    total_inference_time = 0.0
    total_images_processed = 0

    for img_path in image_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, f"{base_name}.txt")

        # Read image to get dimensions
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        # Read Ground Truth
        gt_boxes_by_class = defaultdict(list)
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        if 0 <= cls_id < len(IDD_CLASSES):
                            xc, yc, bw, bh = [float(x) for x in parts[1:5]]
                            x1 = (xc - bw / 2) * w
                            y1 = (yc - bh / 2) * h
                            x2 = (xc + bw / 2) * w
                            y2 = (yc + bh / 2) * h
                            gt_boxes_by_class[cls_id].append([x1, y1, x2, y2])
                            gt_counts[cls_id] += 1

        # Run Model Inference
        t0 = time.time()
        results = model.predict(img, conf=conf_thresh, imgsz=imgsz, verbose=False)[0]
        total_inference_time += (time.time() - t0)
        total_images_processed += 1

        preds_by_class = defaultdict(list)
        if results.boxes is not None:
            for box in results.boxes:
                raw_cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()

                if is_coco:
                    # Map COCO class to IDD class
                    if raw_cls in COCO_TO_IDD:
                        idd_cls = COCO_TO_IDD[raw_cls]
                        preds_by_class[idd_cls].append({"box": xyxy, "conf": conf})
                else:
                    if 0 <= raw_cls < len(IDD_CLASSES):
                        preds_by_class[raw_cls].append({"box": xyxy, "conf": conf})

        # Match predictions to GT for each class
        for cls_id in range(len(IDD_CLASSES)):
            gts = gt_boxes_by_class[cls_id]
            preds = preds_by_class[cls_id]
            # Sort predictions by confidence descending
            preds.sort(key=lambda x: x["conf"], reverse=True)

            matched_gt = set()
            for pred in preds:
                best_iou = 0.0
                best_gt_idx = -1
                for gt_idx, gt_box in enumerate(gts):
                    if gt_idx in matched_gt:
                        continue
                    iou = compute_iou(pred["box"], gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= iou_thresh and best_gt_idx >= 0:
                    matched_gt.add(best_gt_idx)
                    class_preds[cls_id].append({"conf": pred["conf"], "is_tp": 1})
                else:
                    class_preds[cls_id].append({"conf": pred["conf"], "is_tp": 0})

    # Compute AP, Precision, Recall per class
    metrics_per_class = {}
    for cls_id in range(len(IDD_CLASSES)):
        cls_name = IDD_CLASSES[cls_id]
        total_gt = gt_counts[cls_id]
        preds = class_preds[cls_id]

        if total_gt == 0:
            metrics_per_class[cls_name] = {
                "gt_count": total_gt,
                "preds_count": len(preds),
                "precision": None,
                "recall": None,
                "f1": None,
                "ap50": None,
            }
            continue

        preds.sort(key=lambda x: x["conf"], reverse=True)
        tp_cum = np.cumsum([p["is_tp"] for p in preds])
        fp_cum = np.cumsum([1 - p["is_tp"] for p in preds])

        recalls = tp_cum / total_gt
        precisions = tp_cum / (tp_cum + fp_cum) if preds else np.array([])

        ap50 = calculate_ap(recalls, precisions) if preds else 0.0
        operating_threshold = (operating_conf_thresholds or {}).get(
            cls_name, operating_default_threshold
        )
        operating_preds = [p for p in preds if p["conf"] >= operating_threshold]
        operating_tp = sum(p["is_tp"] for p in operating_preds)
        final_precision = operating_tp / len(operating_preds) if operating_preds else 0.0
        final_recall = operating_tp / total_gt
        f1 = (
            2 * (final_precision * final_recall) / (final_precision + final_recall)
            if (final_precision + final_recall) > 0
            else 0.0
        )

        metrics_per_class[cls_name] = {
            "gt_count": total_gt,
            "preds_count": len(preds),
            "precision": final_precision,
            "recall": final_recall,
            "f1": f1,
            "ap50": ap50,
        }

    avg_fps = (
        total_images_processed / total_inference_time if total_inference_time > 0 else 0.0
    )
    avg_latency_ms = (
        (total_inference_time / total_images_processed) * 1000
        if total_images_processed > 0
        else 0.0
    )

    return metrics_per_class, {"fps": avg_fps, "latency_ms": avg_latency_ms, "images": total_images_processed}


def run_benchmark(
    dataset_dir="IDDDetectionsYOLODataset",
    coco_weights="yolov8n.pt",
    idd_weights="models/detection/best.pt",
    split="val",
    num_images=200,
    out_dir="outputs/evaluation/m4_benchmark",
    iou_thresh=IOU_THRESHOLD,
    imgsz=640,
    operating_conf_thresholds=None,
    operating_default_threshold=OPERATING_CONF_THRESHOLD,
):
    os.makedirs(out_dir, exist_ok=True)

    img_dir = os.path.join(dataset_dir, split, "images")
    label_dir = os.path.join(dataset_dir, split, "labels")

    all_images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png")))
    if not all_images:
        print(f"Error: No images found in {img_dir}")
        sys.exit(1)

    if str(num_images).lower() != "all":
        num_images = int(num_images)
        if num_images < 1:
            raise ValueError("num_images must be a positive integer or 'all'")
        # Select evenly spaced images for representative coverage
        step = max(1, len(all_images) // num_images)
        image_subset = all_images[::step][:num_images]
    else:
        image_subset = all_images

    print(f"============================================================")
    print(f"   ASTRA ADAS — M4 BENCHMARK EVALUATION (IDD vs COCO)")
    print(f"============================================================")
    print(f"{split.title()} images to evaluate: {len(image_subset)} (out of {len(all_images)})")
    print(f"COCO Baseline weights:        {coco_weights}")
    print(f"IDD Fine-tuned weights:       {idd_weights}")
    print(f"Inference image size:         {imgsz}")
    print(f"Results output directory:     {out_dir}")
    print(f"------------------------------------------------------------\n")

    # 1. Evaluate COCO Baseline
    print("Evaluating Baseline: YOLOv8n (COCO-pretrained with class mapping)...")
    coco_model = YOLO(coco_weights)
    coco_metrics, coco_timing = evaluate_model(
        coco_model, image_subset, label_dir, is_coco=True,
        iou_thresh=iou_thresh, imgsz=imgsz,
    )
    print(f"  ✓ Baseline evaluated: {coco_timing['fps']:.1f} FPS ({coco_timing['latency_ms']:.1f} ms/image)")

    # 2. Evaluate IDD Fine-Tuned Model
    print("\nEvaluating Fine-Tuned: ASTRA ADAS (IDD-D fine-tuned weights)...")
    idd_model = YOLO(idd_weights)
    idd_metrics, idd_timing = evaluate_model(
        idd_model, image_subset, label_dir, is_coco=False,
        iou_thresh=iou_thresh, imgsz=imgsz,
        operating_conf_thresholds=operating_conf_thresholds,
        operating_default_threshold=operating_default_threshold,
    )
    print(f"  ✓ IDD Fine-tuned evaluated: {idd_timing['fps']:.1f} FPS ({idd_timing['latency_ms']:.1f} ms/image)")

    # 3. Assemble Comparison Table
    records = []
    for cls_name in IDD_CLASSES:
        gt_cnt = idd_metrics[cls_name]["gt_count"]
        c_ap = coco_metrics[cls_name]["ap50"]
        i_ap = idd_metrics[cls_name]["ap50"]
        if gt_cnt:
            delta_ap = i_ap - c_ap
            pct_gain = (
                ((i_ap - c_ap) / max(c_ap, 1e-4)) * 100
                if c_ap > 0
                else (100.0 if i_ap > 0 else 0.0)
            )
        else:
            delta_ap = pct_gain = np.nan

        records.append({
            "class": cls_name,
            "category": "Indian-Specific" if cls_name in INDIAN_SPECIFIC_CLASSES else "Standard",
            "gt_count": gt_cnt,
            "coco_baseline_ap50": round(c_ap, 4) if c_ap is not None else np.nan,
            "idd_finetuned_ap50": round(i_ap, 4) if i_ap is not None else np.nan,
            "delta_ap50": round(delta_ap, 4) if not np.isnan(delta_ap) else np.nan,
            "gain_pct": round(pct_gain, 1) if not np.isnan(pct_gain) else np.nan,
            "finetuned_precision": (
                round(idd_metrics[cls_name]["precision"], 3)
                if idd_metrics[cls_name]["precision"] is not None else np.nan
            ),
            "finetuned_recall": (
                round(idd_metrics[cls_name]["recall"], 3)
                if idd_metrics[cls_name]["recall"] is not None else np.nan
            ),
            "finetuned_f1": (
                round(idd_metrics[cls_name]["f1"], 3)
                if idd_metrics[cls_name]["f1"] is not None else np.nan
            ),
        })

    df = pd.DataFrame(records)
    csv_name = f"m4_model_comparison_{split}_{imgsz}.csv"
    csv_path = os.path.join(out_dir, csv_name)
    df.to_csv(csv_path, index=False)
    print(f"\nDetailed per-class comparison saved to {csv_path}")

    # Aggregated mAPs
    evaluated_classes = df[df["gt_count"] > 0]
    coco_map50 = evaluated_classes["coco_baseline_ap50"].mean()
    idd_map50 = evaluated_classes["idd_finetuned_ap50"].mean()

    df_indian = evaluated_classes[evaluated_classes["category"] == "Indian-Specific"]
    coco_indian_map = df_indian["coco_baseline_ap50"].mean()
    idd_indian_map = df_indian["idd_finetuned_ap50"].mean()

    df_standard = evaluated_classes[evaluated_classes["category"] == "Standard"]
    coco_std_map = df_standard["coco_baseline_ap50"].mean()
    idd_std_map = df_standard["idd_finetuned_ap50"].mean()

    # 4. Generate Plot: AP@50 Comparison Bar Chart
    plt.figure(figsize=(14, 7))
    x = np.arange(len(df))
    width = 0.38

    plt.bar(x - width/2, df["coco_baseline_ap50"], width, label="COCO Baseline (yolov8n)", color="#94a3b8")
    plt.bar(x + width/2, df["idd_finetuned_ap50"], width, label="IDD Fine-Tuned (Astra ADAS)", color="#2563eb")

    # Add asterisk / highlight to Indian-specific classes
    labels = []
    for cls_name in df["class"]:
        if cls_name in INDIAN_SPECIFIC_CLASSES:
            labels.append(f"{cls_name}*")
        else:
            labels.append(cls_name)

    plt.xlabel("Object Class (* = Indian-Specific Road User)", fontsize=11, fontweight="bold")
    plt.ylabel("Average Precision @ IoU 0.50 (AP@50)", fontsize=11, fontweight="bold")
    plt.title(f"Astra ADAS (M4) — Detection Benchmark on Indian Driving Dataset ({len(image_subset)} {split} Images)\n"
              f"Overall mAP@50: Baseline {coco_map50:.1%} → Fine-Tuned {idd_map50:.1%} (+{(idd_map50-coco_map50)*100:+.1f}%)",
              fontsize=13, fontweight="bold", pad=15)
    plt.xticks(x, labels, rotation=45, ha="right", fontsize=10)
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()

    plot_path = os.path.join(out_dir, "m4_class_ap_comparison.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Comparison chart saved to {plot_path}")

    # 5. Generate Markdown Report
    import torch
    hardware = (
        f"CUDA GPU ({torch.cuda.get_device_name(0)})"
        if torch.cuda.is_available()
        else f"CPU ({os.cpu_count()} logical threads)"
    )
    operating_thresholds = {"default": operating_default_threshold}
    operating_thresholds.update(operating_conf_thresholds or {})
    operating_threshold_text = ", ".join(
        f"{name}: {threshold:.3f}"
        for name, threshold in operating_thresholds.items()
    )
    md_content = f"""# Astra ADAS — Milestone M4 Detection Benchmark Report

**Evaluation Split:** IDD-D (`{split}`) | **Images Evaluated:** {len(image_subset)} / {len(all_images)}<br>
**Hardware Executed:** {hardware}<br>
**Inference settings:** image size {imgsz}px; AP confidence floor {AP_CONF_THRESHOLD}; IoU threshold {iou_thresh}; 101-point interpolation<br>
**IDD Precision/Recall/F1 settings:** class confidence thresholds ({operating_threshold_text})

---

## 1. Executive Summary & Key Results

| Metric | COCO Baseline (`yolov8n.pt`) | IDD Fine-Tuned (`models/detection/best.pt`) | Absolute Improvement | Relative Gain |
|---|:---:|:---:|:---:|:---:|
| **Overall mAP@50 ({len(evaluated_classes)} classes with ground truth)** | **{coco_map50:.3f}** | **{idd_map50:.3f}** | **{idd_map50 - coco_map50:+.3f}** | **{((idd_map50 - coco_map50)/max(coco_map50, 1e-4))*100:+.1f}%** |
| **Indian-Specific Classes mAP@50** | **{coco_indian_map:.3f}** | **{idd_indian_map:.3f}** | **{idd_indian_map - coco_indian_map:+.3f}** | **{((idd_indian_map - coco_indian_map)/max(coco_indian_map, 1e-4))*100:+.1f}%** |
| **Standard Shared Classes mAP@50** | **{coco_std_map:.3f}** | **{idd_std_map:.3f}** | **{idd_std_map - coco_std_map:+.3f}** | **{((idd_std_map - coco_std_map)/max(coco_std_map, 1e-4))*100:+.1f}%** |
| **Average Inference Latency** | {coco_timing['latency_ms']:.1f} ms | {idd_timing['latency_ms']:.1f} ms | — | — |
| **Throughput (Inference FPS)** | {coco_timing['fps']:.1f} FPS | {idd_timing['fps']:.1f} FPS | — | — |

---

## 2. Per-Class Benchmark Comparison Table

> `*` indicates Indian-specific classes not native to standard international datasets like MS COCO. Classes with no ground-truth instances in this split have undefined AP/P/R/F1 and are excluded from mAP averages.

| Class | Category | Ground Truth Count | Baseline AP@50 | IDD Fine-Tuned AP@50 | Delta AP@50 | Fine-Tuned Precision | Fine-Tuned Recall | Fine-Tuned F1 |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    for _, row in df.iterrows():
        is_star = "*" if row["class"] in INDIAN_SPECIFIC_CLASSES else ""
        md_content += (
            f"| `{row['class']}{is_star}` | {row['category']} | {row['gt_count']} | "
            f"{row['coco_baseline_ap50']:.3f} | **{row['idd_finetuned_ap50']:.3f}** | "
            f"`{row['delta_ap50']:+.3f}` | {row['finetuned_precision']:.3f} | "
            f"{row['finetuned_recall']:.3f} | {row['finetuned_f1']:.3f} |\n"
        )

    md_content += f"""
---

## 3. Interpretation

These metrics describe this split and evaluation configuration only. They do not establish calibrated distance/TTC performance, tracking accuracy, or road safety. The IDD checkpoint should be considered better on detection only if its mAP@50 exceeds the baseline on the same held-out split.

*(Generated by `src.evaluation.benchmark_m4`)*
"""

    report_path = os.path.join(out_dir, "m4_benchmark_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Benchmark Markdown report saved to {report_path}")

    return df, report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Astra ADAS M4 Quantitative Benchmark")
    parser.add_argument("--num-images", default="all", help="Number of validation images to test (e.g. 150, 300, or 'all'). Default: all")
    parser.add_argument("--split", default="val", choices=("val", "test"), help="Dataset split")
    parser.add_argument("--out-dir", default="outputs/evaluation/m4_benchmark", help="Output directory")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Inference image size in pixels (e.g. 640 or 960)")
    parser.add_argument("--config", default="config.yaml",
                        help="Config file providing the IDD per-class operating thresholds")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    run_benchmark(
        num_images=args.num_images,
        split=args.split,
        out_dir=args.out_dir,
        imgsz=args.imgsz,
        operating_conf_thresholds=config["detection"].get(
            "class_conf_thresholds", {}
        ),
        operating_default_threshold=config["detection"]["conf_threshold"],
    )
