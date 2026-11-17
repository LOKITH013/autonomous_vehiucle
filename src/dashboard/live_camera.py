import cv2


def draw_live_detections(frame, detections):
    height, width = frame.shape[:2]
    detected_classes = sorted({detection["cls_name"] for detection in detections})
    summary = f"Objects: {len(detections)}"
    if detected_classes:
        summary += " | " + ", ".join(detected_classes[:5])
        if len(detected_classes) > 5:
            summary += ", ..."
    cv2.putText(
        frame, summary, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
        (0, 255, 0), 2, cv2.LINE_AA,
    )

    for detection in detections:
        x1, y1, x2, y2 = (
            int(round(coordinate)) for coordinate in detection["xyxy"]
        )
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{detection['cls_name']} {detection['conf']:.0%}"
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        label_top = max(0, y1 - label_height - baseline - 6)
        cv2.rectangle(
            frame, (x1, label_top),
            (min(width - 1, x1 + label_width + 8), y1),
            (0, 255, 0), cv2.FILLED,
        )
        cv2.putText(
            frame, label, (x1 + 4, min(y1 - 2, label_top + label_height + 2)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
        )

    return frame
