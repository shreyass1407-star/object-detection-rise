🔍 Real-Time Object Detection System using YOLOv8
RISE Internship — Deep Learning & Neural Networks (Project 4)
---
📁 Project Structure
```
object_detection_project/
│
├── detect.py              ← Main script (run this)
├── requirements.txt       ← All dependencies
├── README.md              ← This file
│
├── sample_images/         ← Put your test images here
│   └── test.jpg           ← (auto-downloaded if missing)
│
└── outputs/               ← All results saved here
    ├── test_detected.png
    ├── batch_summary.png
    └── class_distribution.png
```
---
⚙️ Setup (One-Time)
Step 1 — Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```
Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```
> ⚠️ This installs PyTorch + YOLOv8. It may take a few minutes on first run.
Step 3 — Run the project
```bash
python detect.py
```
> On the **very first run**, YOLOv8 will automatically download the
> `yolov8n.pt` model weights (~6 MB). Internet connection required.
---
🚀 Usage Modes
Open `detect.py` and scroll to the `main()` function at the bottom.
You will see three options — just uncomment the one you want:
Option A — Single Image (default, always active)
Place any `.jpg` or `.png` inside `sample_images/` and name it `test.jpg`.
The script detects objects and saves a side-by-side result to `outputs/`.
Option B — Batch (entire folder)
```python
detect_folder(model, "sample_images")
```
Runs detection on every image in the folder and saves a summary bar chart.
Option C — Live Webcam
```python
detect_webcam(model, camera_index=0)
```
Opens your webcam, draws live bounding boxes with FPS counter.
Press Q to quit.
---
🎛️ Tweaking Parameters
Edit the `CONFIG` dict at the top of `detect.py`:
Parameter	Default	What it does
`model_name`	`yolov8n.pt`	Model size: n/s/m/l/x (larger = more accurate, slower)
`confidence`	`0.40`	Min confidence to show a detection (0.0–1.0)
`iou_threshold`	`0.45`	Controls overlap filtering (lower = fewer boxes)
`input_size`	`640`	Image resolution fed to model
Model size guide:
`yolov8n.pt` — Nano (fastest, good for laptops)
`yolov8s.pt` — Small (balanced)
`yolov8m.pt` — Medium (more accurate)
`yolov8l.pt` — Large
`yolov8x.pt` — Extra-large (best accuracy, needs GPU)
---
📊 What the Script Produces
Output File	Description
`outputs/<name>_detected.png`	Side-by-side original vs detected image
`outputs/batch_summary.png`	Bar chart of detections per image (batch mode)
`outputs/class_distribution.png`	Horizontal bar chart of detected object classes
---
🗂️ COCO Classes (what YOLOv8 can detect)
YOLOv8 is pre-trained on the COCO dataset — 80 common object categories:
> person, bicycle, car, motorcycle, airplane, bus, train, truck, boat,
> traffic light, fire hydrant, stop sign, parking meter, bench, bird,
> cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack,
> umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball,
> kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket,
> bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple,
> sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair,
> couch, potted plant, bed, dining table, toilet, tv, laptop, mouse,
> remote, keyboard, cell phone, microwave, oven, toaster, sink,
> refrigerator, book, clock, vase, scissors, teddy bear, hair drier,
> toothbrush  *(80 total)*
---
🧠 How It Works (Brief)
```
Input Image
    │
    ▼
YOLOv8 Backbone (CSPDarknet)
    │   Extracts feature maps at multiple scales
    ▼
YOLOv8 Neck (PAN-FPN)
    │   Fuses multi-scale features
    ▼
YOLOv8 Head
    │   Predicts bounding boxes + class probabilities
    ▼
Non-Maximum Suppression (NMS)
    │   Removes overlapping duplicate boxes
    ▼
Annotated Output Image
```
---
🛠️ Troubleshooting
Problem	Fix
`ModuleNotFoundError: ultralytics`	Run `pip install -r requirements.txt`
Webcam not opening	Try `camera_index=1` or check camera permissions
Slow inference	Use `yolov8n.pt` (nano) or reduce `input_size` to 320
No GPU detected	Install CUDA-enabled PyTorch from pytorch.org
Model not downloading	Check internet connection; proxy may block downloads
---
👨‍💻 Internship Info
Program : RISE Internship — Tamizan Skills
Domain  : Deep Learning & Neural Networks
Project : 4 — Industry-Oriented Object Detection
Contact : Rise@tamizhanskills.com | +91 6383418100