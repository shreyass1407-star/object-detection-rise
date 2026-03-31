"""
====================================================
 Real-Time Object Detection System using YOLOv8
 RISE Internship - Deep Learning & Neural Networks
 Project 4: Industry-Oriented Object Detection
====================================================
"""

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from ultralytics import YOLO
import time
import os

#  CONFIGURATION
CONFIG = {
    "model_name"      : "yolov8n.pt",   # nano model – fast & lightweight
    "confidence"      : 0.40,           # minimum confidence threshold
    "iou_threshold"   : 0.45,           # IoU threshold for NMS
    "input_size"      : 640,            # image input size for YOLO
    "output_dir"      : "outputs",      # folder to save results
    "sample_images_dir": "sample_images",
}

# Create output directory
Path(CONFIG["output_dir"]).mkdir(exist_ok=True)
Path(CONFIG["sample_images_dir"]).mkdir(exist_ok=True)

#  80 COCO CLASS COLORS  (one per class)
np.random.seed(42)
CLASS_COLORS = {i: tuple(np.random.randint(50, 255, 3).tolist()) for i in range(80)}

#  HELPER: draw bounding boxes on frame
def draw_detections(frame, results, model):
    """Draw bounding boxes and labels on the image frame."""
    annotated = frame.copy()
    detection_info = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf  = float(box.conf[0])
            cls   = int(box.cls[0])
            label = model.names[cls]
            color = CLASS_COLORS[cls % len(CLASS_COLORS)]

            # Draw filled rectangle header
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            text  = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            detection_info.append({"label": label, "confidence": conf,
                                   "bbox": (x1, y1, x2, y2)})

    return annotated, detection_info

#  FUNCTION 1 – Detect on a single image file
def detect_image(model, image_path: str, save: bool = True):
    """Run detection on a single image and display/save results."""
    print(f"\n[INFO] Running detection on: {image_path}")

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    t0 = time.time()
    results = model.predict(
        source     = img_bgr,
        conf       = CONFIG["confidence"],
        iou        = CONFIG["iou_threshold"],
        imgsz      = CONFIG["input_size"],
        verbose    = False,
    )
    inference_ms = (time.time() - t0) * 1000

    annotated_bgr, detections = draw_detections(img_bgr, results, model)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    # ── Matplotlib figure ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#0d0d0d")

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image", color="white", fontsize=13, pad=10)
    axes[0].axis("off")

    axes[1].imshow(annotated_rgb)
    axes[1].set_title(
        f"Detections  |  {len(detections)} object(s)  |  {inference_ms:.1f} ms",
        color="white", fontsize=13, pad=10,
    )
    axes[1].axis("off")

    plt.suptitle("YOLOv8 Object Detection — RISE Internship",
                 color="#00d4ff", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save:
        stem = Path(image_path).stem
        out  = os.path.join(CONFIG["output_dir"], f"{stem}_detected.png")
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
        print(f"[SAVED] {out}")

    plt.show()

    # ── Print detection table ──
    print(f"\n{'─'*50}")
    print(f"  {'Object':<20} {'Confidence':>12}   {'Bounding Box'}")
    print(f"{'─'*50}")
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        print(f"  {d['label']:<20} {d['confidence']:>11.2%}   ({x1},{y1}) → ({x2},{y2})")
    if not detections:
        print("  No objects detected above threshold.")
    print(f"{'─'*50}")
    print(f"  Inference time : {inference_ms:.1f} ms")
    print(f"{'─'*50}\n")

    return detections

#  FUNCTION 2 – Detect on a folder of images
def detect_folder(model, folder_path: str):
    """Run detection on every image inside a folder."""
    exts   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in Path(folder_path).iterdir() if p.suffix.lower() in exts]

    if not images:
        print(f"[WARN] No images found in: {folder_path}")
        return

    print(f"\n[INFO] Detecting in {len(images)} image(s) from '{folder_path}' …")
    summary = []

    for img_path in images:
        dets = detect_image(model, str(img_path), save=True)
        if dets is not None:
            summary.append({"file": img_path.name, "count": len(dets)})

    # ── Summary bar chart ──
    if summary:
        names  = [s["file"] for s in summary]
        counts = [s["count"] for s in summary]

        fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.4), 5))
        fig.patch.set_facecolor("#0d0d0d")
        ax.set_facecolor("#141414")

        bars = ax.bar(names, counts, color="#00d4ff", edgecolor="#005f73", linewidth=1.2)
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    str(cnt), ha="center", va="bottom", color="white", fontsize=11)

        ax.set_xlabel("Image", color="white")
        ax.set_ylabel("Detections", color="white")
        ax.set_title("Object Count per Image", color="#00d4ff", fontsize=14)
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#333")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()

        out = os.path.join(CONFIG["output_dir"], "batch_summary.png")
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
        print(f"[SAVED] Batch summary chart → {out}")
        plt.show()

#  FUNCTION 3 – Webcam / live detection
def detect_webcam(model, camera_index: int = 0, max_frames: int = 300):
    """
    Run live YOLOv8 detection on your webcam.
    Press  Q  to quit early.
    max_frames  limits recording to avoid accidental infinite loops.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check camera_index or permissions.")
        return

    print(f"\n[INFO] Webcam opened (index={camera_index}). Press Q to quit …\n")
    frame_count = 0
    fps_display = 0
    t_prev = time.time()

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed.")
            break

        results = model.predict(
            source  = frame,
            conf    = CONFIG["confidence"],
            iou     = CONFIG["iou_threshold"],
            imgsz   = CONFIG["input_size"],
            verbose = False,
        )

        annotated, _ = draw_detections(frame, results, model)

        # FPS counter
        t_now       = time.time()
        fps_display = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev      = t_now

        cv2.putText(annotated, f"FPS: {fps_display:.1f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 212, 255), 2)
        cv2.putText(annotated, "Press Q to quit", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("YOLOv8 Real-Time Detection — RISE Internship", annotated)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Quit by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Webcam detection stopped after {frame_count} frames.")

#  FUNCTION 4 – Class distribution analysis
def analyse_detections(detections_list: list, title: str = "Detection Analysis"):
    """Plot class frequency from a list of detection results."""
    from collections import Counter

    all_labels = [d["label"] for dets in detections_list for d in dets]
    if not all_labels:
        print("[INFO] No detections to analyse.")
        return

    counter = Counter(all_labels)
    labels  = [k for k, _ in counter.most_common()]
    counts  = [counter[k] for k in labels]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 5))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#141414")

    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(labels)))
    bars   = ax.barh(labels[::-1], counts[::-1], color=colors[::-1], edgecolor="#222")

    for bar, cnt in zip(bars, counts[::-1]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                str(cnt), va="center", color="white", fontsize=10)

    ax.set_xlabel("Count", color="white")
    ax.set_title(title, color="#00d4ff", fontsize=14)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333")
    plt.tight_layout()

    out = os.path.join(CONFIG["output_dir"], "class_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    print(f"[SAVED] Class distribution → {out}")
    plt.show()

#  MAIN
def main():
    print("=" * 60)
    print("  Real-Time Object Detection System — YOLOv8")
    print("  RISE Internship | Deep Learning & Neural Networks")
    print("=" * 60)

    # ── Load model (downloads automatically on first run) ──
    print(f"\n[INFO] Loading model: {CONFIG['model_name']} …")
    model = YOLO(CONFIG["model_name"])
    print(f"[INFO] Model loaded. Classes: {len(model.names)}")
    print(f"[INFO] Device: {'CUDA (GPU)' if torch.cuda.is_available() else 'CPU'}\n")

    #  CHOOSE WHAT TO RUN — edit the lines below as needed

    # OPTION A ─ Detect on a SINGLE image
    # Place any .jpg/.png file in the project folder and update the path:
    sample_img = "sample_images/test.jpg"
    if Path(sample_img).exists():
        dets = detect_image(model, sample_img)
        if dets:
            analyse_detections([dets], title="Single Image — Class Distribution")
    else:
        print(f"[INFO] '{sample_img}' not found.")
        print("       Add a test image to sample_images/ folder and re-run.\n")
        # ── Demo with a synthetically generated test image ──
        _demo_with_synthetic(model)

    # OPTION B ─ Detect on ALL images in a folder  (uncomment to use)
    # detect_folder(model, "sample_images")

    # OPTION C ─ Live webcam detection  (uncomment to use)
    # detect_webcam(model, camera_index=0)

    print("\n[DONE] All results saved to the 'outputs/' folder.")

#  DEMO fallback – creates a quick test image
#  from a public URL so the project always runs
def _demo_with_synthetic(model):
    """Download a free sample image and run detection on it."""
    import urllib.request

    url      = "https://ultralytics.com/images/bus.jpg"
    out_path = "sample_images/test.jpg"

    print(f"[INFO] Downloading demo image from Ultralytics …")
    try:
        urllib.request.urlretrieve(url, out_path)
        print(f"[INFO] Saved demo image → {out_path}")
        dets = detect_image(model, out_path)
        if dets:
            analyse_detections([dets], title="Demo Image — Class Distribution")
    except Exception as e:
        print(f"[WARN] Could not download demo image: {e}")
        print("       Please add your own image to sample_images/test.jpg and re-run.")


if __name__ == "__main__":
    main()
