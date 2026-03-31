# backend/back/pipeline.py
import os
import cv2
import zipfile
import importlib.util
import json

# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(BASE_DIR, "saved_models", "final_vastu.zip")
EXTRACT_DIR = os.path.join(BASE_DIR, "saved_models", "vastu")

# ---------------- Extract ZIP if not already ----------------
if not os.path.exists(EXTRACT_DIR):
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    print("✅ Models extracted from ZIP")

# Paths to models
YOLO_PATH = os.path.join(EXTRACT_DIR, "yolov8n.pt")
PY_MODEL_PATH = os.path.join(EXTRACT_DIR, "final", "vastu_3d_full.py")  # Make sure your Python file has a valid name

print("YOLO_PATH:", YOLO_PATH, "Exists:", os.path.exists(YOLO_PATH))
print("PY_MODEL_PATH:", PY_MODEL_PATH, "Exists:", os.path.exists(PY_MODEL_PATH))

# ---------------- Load YOLO Model ----------------
yolo_model = None
def get_yolo():
    global yolo_model
    if yolo_model is None:
        if not os.path.exists(YOLO_PATH):
            raise Exception(f"❌ YOLO model NOT FOUND at: {YOLO_PATH}")
        from ultralytics import YOLO
        yolo_model = YOLO(YOLO_PATH)
    return yolo_model

# ---------------- Load Vastu Python Model ----------------
vastu_module = None
def get_vastu_module():
    global vastu_module
    if vastu_module is None:
        if not os.path.exists(PY_MODEL_PATH):
            raise Exception(f"❌ Vastu Python model NOT FOUND at: {PY_MODEL_PATH}")
        spec = importlib.util.spec_from_file_location("vastu_3d_full.py", PY_MODEL_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        vastu_module = module
    return vastu_module

# ---------------- Frame Extraction ----------------
def extract_frames(video_path, max_frames=50):
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Cannot open video file!")

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret or count >= max_frames:
            break
        frames.append(frame)
        count += 1
    cap.release()
    print(f"✅ Extracted {len(frames)} frames")
    return frames

# ---------------- Furniture Detection ----------------
def detect_furniture(frames):
    model = get_yolo()
    detected_items = {}
    for idx, frame in enumerate(frames):
        results = model(frame)
        for box in results[0].boxes:
            cls = box.cls[0].item()  # class index
            detected_items[cls] = detected_items.get(cls, 0) + 1
    print(f"✅ Detected furniture: {detected_items}")
    return detected_items

# ---------------- Vastu/Optimization/Genetics Analysis ----------------
def run_vastu_analysis(detected_items, room_type, furniture_data):
    module = get_vastu_module()
    if isinstance(furniture_data, str):
        furniture_data = json.loads(furniture_data)
    combined_data = {**detected_items, **furniture_data}
    
    # Make sure your Python model has a function called run_vastu_analysis
    analysis_result = module.run_vastu_analysis(combined_data, room_type)
    return analysis_result

# ---------------- Full Pipeline ----------------
def run_full_pipeline(video_path, room_type, furniture_data):
    print("🚀 PIPELINE STARTED")
    frames = extract_frames(video_path)
    detected_items = detect_furniture(frames)
    analysis_result = run_vastu_analysis(detected_items, room_type, furniture_data)
    
    return {
        "detected_furniture": detected_items,
        "analysis_result": analysis_result,
        "video_url": f"/media/videos/{os.path.basename(video_path)}"
    }