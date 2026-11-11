# -*- coding: utf-8 -*-
"""pcb_final_submission.ipynb
"""

# ==========================================
# 1. INSTALL & DOWNLOAD DATASET (Roboflow)
# ==========================================
!pip install -q roboflow ultralytics
from roboflow import Roboflow
rf = Roboflow(api_key="ActMIyshXhcAxSp1TVna")
project = rf.workspace("uthm-project-visionbased-pcb-defect-detection").project("pcb-defect-detection-9ewqw-duvuw")
version = project.version(2)
dataset = version.download("yolov8")


# ==========================================
# 2. TRAIN YOLOv8m MODEL
# ==========================================
from ultralytics import YOLO

model = YOLO("yolov8m.pt")
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
    optimizer='AdamW',
    lr0=0.001,
    patience=15,
    project="pcb_hackathon",
    name="best_yolov8m",
    save=True
)


# ==========================================
# 3. UNZIP TEST DATASET
# ==========================================
!unzip -q /content/test.zip -d /content/


# ==========================================
# 4. EVALUATE MODEL ON TEST LABELS (Check mAP)
# ==========================================
import yaml

test_yaml = {
    'path': '/content/test',
    'train': 'images',
    'val': 'images',
    'names': model.names
}

with open('/content/test_evaluate.yaml', 'w') as f:
    yaml.dump(test_yaml, f)

metrics = model.val(data='/content/test_evaluate.yaml')
print(f"\n🏆 Test mAP50: {metrics.box.map50 * 100:.2f}% | mAP50-95: {metrics.box.map * 100:.2f}%")


# ==========================================
# 5. FINAL OPTIMIZED INFERENCE & SUBMISSION CSV
# ==========================================
import os
import glob
import pandas as pd
from google.colab import files

# Load trained weights
model = YOLO('/content/runs/detect/pcb_hackathon/best_yolov8m/weights/best.pt')

test_images_dir = '/content/test/images'
image_paths = glob.glob(os.path.join(test_images_dir, '*.*'))

# Advanced Class Size Limits (Normalized width/height limits)
class_size_limits = {
    0: {'max_w': 0.15, 'max_h': 0.15},  # open_circuit
    1: {'max_w': 0.18, 'max_h': 0.18},  # short
    2: {'max_w': 0.12, 'max_h': 0.12},  # spur / spurious copper
    3: {'max_w': 0.10, 'max_h': 0.10}   # mouse_bite
}

submission_rows = []

print("Running optimized inference...")
for img_path in image_paths:
    filename = os.path.basename(img_path)
    
    # Run prediction with TTA and low confidence threshold
    results = model.predict(source=img_path, conf=0.001, iou=0.6, augment=True, verbose=False)
    
    for r in results:
        boxes = r.boxes
        if len(boxes) == 0:
            continue
        
        coords = boxes.xyxy.cpu().numpy()
        norm_boxes = boxes.xywhn.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        
        for box, norm_box, conf, cls_idx in zip(coords, norm_boxes, confs, classes):
            norm_w, norm_h = norm_box[2], norm_box[3]
            
            # Apply class-specific size heuristic
            if cls_idx in class_size_limits:
                limit = class_size_limits[cls_idx]
                if norm_w > limit['max_w'] or norm_h > limit['max_h']:
                    continue
            
            x_min, y_min, x_max, y_max = box
            x_min, y_min = max(0, x_min), max(0, y_min)
            
            submission_rows.append({
                'image_name': filename,
                'class_name': model.names[cls_idx],
                'x_min': round(float(x_min), 2),
                'y_min': round(float(y_min), 2),
                'x_max': round(float(x_max), 2),
                'y_max': round(float(y_max), 2),
                'confidence': round(float(conf), 4)
            })

df = pd.DataFrame(submission_rows)
df.to_csv('submission.csv', index=False)
print(f"Success! Generated {len(submission_rows)} predictions.")
files.download('submission.csv')