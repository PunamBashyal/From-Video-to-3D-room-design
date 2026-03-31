# --------------------------
# SERVER-SAFE BASE CONFIG (First, to avoid GUI crashes)
# --------------------------
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cloud/servers
import os, json, uuid, logging, pickle, datetime, random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from collections import defaultdict, Counter
import torch
import trimesh
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

YOLO_MODEL      = None
MIDAS_MODEL     = None
MIDAS_TRANSFORM = None
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------------------
# CONSTANTS — TUNED FOR MAX DETECTION
# --------------------------
SUPPORTED_EXTS  = ('.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v')
FRAME_SAMPLE_RATE = 5      # sample every 5th frame (was 20) → 4× more frames analysed
GA_GENERATIONS  = 100
GA_POPULATION   = 50
CAMERA_FOV      = 60

# Expanded object set — covers everything YOLOv8 can detect indoors
ROOM_OBJECTS = {
    # Sleeping / seating
    'bed', 'sofa', 'couch', 'chair', 'bench', 'stool',
    # Tables & storage
    'dining table', 'table', 'desk', 'cabinet', 'bookcase',
    'shelf', 'nightstand', 'dresser', 'wardrobe', 'cupboard', 'drawer',
    # Kitchen
    'refrigerator', 'oven', 'microwave', 'sink', 'toaster', 'kettle',
    'dishwasher', 'washing machine', 'dryer',
    # Electronics
    'tv', 'television', 'laptop', 'computer', 'monitor', 'cell phone',
    'keyboard', 'mouse', 'remote',
    # Bathroom
    'toilet', 'bathtub', 'shower', 'towel',
    # Decor / plants
    'potted plant', 'vase', 'clock', 'mirror', 'book',
    'painting', 'picture frame', 'candle', 'flower',
    # Lighting
    'lamp', 'chandelier', 'light',
    # Soft furnishings
    'curtain', 'carpet', 'rug', 'blanket', 'pillow', 'cushion',
    # Misc indoor
    'bicycle', 'umbrella', 'backpack', 'suitcase', 'bag',
    'trash can', 'dustbin', 'bin', 'fan', 'air conditioner',
    'door', 'window',
}

# Full Vastu rules (unchanged)
VASTU_RULES_FULL = {
    'bed':          {'zones': ['South-West', 'South', 'West'],
                     'rationale': 'South-West (Nairutya) is the stabilising Earth corner. Sleeping with head toward South/West aligns with Earth magnetism for deep, restorative sleep. Avoid North.'},
    'sofa':         {'zones': ['North', 'North-East', 'East', 'West'],
                     'rationale': 'Lightweight upholstered seating belongs in the lighter Northern arc.'},
    'couch':        {'zones': ['North', 'North-East', 'East', 'West'],
                     'rationale': 'Same as sofa. Couches should remain in the energetically active Northern arc.'},
    'chair':        {'zones': ['West', 'East', 'North', 'North-West'],
                     'rationale': 'Work chairs face East or North. West is acceptable for reading.'},
    'bench':        {'zones': ['East', 'North', 'West'],
                     'rationale': 'Open benches carry the same energy as chairs.'},
    'stool':        {'zones': ['East', 'North', 'West'],
                     'rationale': 'Stools follow chair placement rules.'},
    'dining table': {'zones': ['West', 'North-West'],
                     'rationale': 'West (Varun) governs nourishment. North-West supports digestion.'},
    'table':        {'zones': ['West', 'North-West', 'East'],
                     'rationale': 'General tables follow dining table or desk rules depending on use.'},
    'desk':         {'zones': ['East', 'North', 'North-East'],
                     'rationale': 'East faces the rising Sun — ideal for study and work desks.'},
    'cabinet':      {'zones': ['South-West', 'South', 'West'],
                     'rationale': 'Heavy storage cabinets must sit in South-West to ground the space.'},
    'bookcase':     {'zones': ['East', 'North-East', 'North'],
                     'rationale': 'Books represent knowledge (Saraswati). North-East and East are zones of intellect.'},
    'shelf':        {'zones': ['East', 'North', 'West'],
                     'rationale': 'Shelves follow cabinet or bookcase rules depending on content.'},
    'nightstand':   {'zones': ['South-West', 'South', 'West'],
                     'rationale': 'Bedside tables accompany the bed — same South-West zone.'},
    'dresser':      {'zones': ['South-West', 'South', 'West'],
                     'rationale': 'Dressers are heavy storage — South-West is ideal.'},
    'wardrobe':     {'zones': ['South-West', 'South', 'West'],
                     'rationale': 'Wardrobes are heavy and should be in South-West like cabinets.'},
    'cupboard':     {'zones': ['South-West', 'South', 'West'],
                     'rationale': 'Same as cabinet — heavy storage belongs in South-West.'},
    'refrigerator': {'zones': ['South-East', 'South', 'West'],
                     'rationale': 'South-East (Agni corner) governs all appliances.'},
    'oven':         {'zones': ['South-East'],
                     'rationale': 'The oven is quintessential Agni (fire). South-East is the fire corner.'},
    'microwave':    {'zones': ['South-East', 'South'],
                     'rationale': 'Fire-element appliance — same as oven.'},
    'sink':         {'zones': ['North-East', 'North', 'East'],
                     'rationale': 'North-East is the Jal (water) zone.'},
    'toaster':      {'zones': ['South-East', 'South'],
                     'rationale': 'Fire-element small appliance.'},
    'kettle':       {'zones': ['South-East', 'South'],
                     'rationale': 'Heating element makes it a fire appliance.'},
    'dishwasher':   {'zones': ['North-East', 'North', 'West'],
                     'rationale': 'Water appliance — follows sink rules.'},
    'washing machine': {'zones': ['North-West', 'West'],
                        'rationale': 'Movement and water — North-West (Vayavya) is ideal.'},
    'dryer':        {'zones': ['South-East', 'North-West'],
                     'rationale': 'Heat appliance — South-East preferred.'},
    'tv':           {'zones': ['East', 'South-East'],
                     'rationale': 'TV faces East so viewers face West (auspicious).'},
    'television':   {'zones': ['East', 'South-East'],
                     'rationale': 'Same as tv.'},
    'laptop':       {'zones': ['East', 'North', 'North-East'],
                     'rationale': 'Portable work device — follows desk rules.'},
    'computer':     {'zones': ['East', 'North'],
                     'rationale': 'Desktop computers represent sustained work energy.'},
    'monitor':      {'zones': ['East', 'North'],
                     'rationale': 'Same as desktop — East/North placement.'},
    'cell phone':   {'zones': ['East', 'North', 'North-East'],
                     'rationale': 'Mobile charging in the knowledge zone.'},
    'keyboard':     {'zones': ['East', 'North'],
                     'rationale': 'Follows computer/desk zone rules.'},
    'toilet':       {'zones': ['West', 'North-West', 'South'],
                     'rationale': 'Toilets must never be in North-East.'},
    'bathtub':      {'zones': ['East', 'North-East', 'North'],
                     'rationale': 'Bathing is a purification ritual — East/North-East.'},
    'shower':       {'zones': ['East', 'North-East', 'North'],
                     'rationale': 'Same as bathtub — water purification zone.'},
    'potted plant': {'zones': ['North-East', 'East', 'North'],
                     'rationale': 'Living plants in North-East and East get morning Sun.'},
    'vase':         {'zones': ['East', 'North-East', 'North'],
                     'rationale': 'Vases hold water and flowers — Jal zone.'},
    'clock':        {'zones': ['North', 'East', 'North-East'],
                     'rationale': 'Clocks represent time flow — North and East are ideal.'},
    'mirror':       {'zones': ['North', 'East'],
                     'rationale': 'Mirrors double energy — North doubles prosperity.'},
    'book':         {'zones': ['East', 'North-East', 'North'],
                     'rationale': 'Books represent knowledge — North-East/East.'},
    'lamp':         {'zones': ['South-East', 'East', 'North-East'],
                     'rationale': 'Lamps represent Agni. South-East is the primary fire corner.'},
    'chandelier':   {'zones': ['Center'],
                     'rationale': 'Chandelier belongs at the Brahmasthan (sacred center).'},
    'light':        {'zones': ['South-East', 'East', 'Center'],
                     'rationale': 'Lighting follows lamp rules.'},
    'curtain':      {'zones': ['East', 'North', 'West'],
                     'rationale': 'Curtains regulate light — East and North should be light-coloured.'},
    'carpet':       {'zones': ['Center', 'East', 'North'],
                     'rationale': 'A carpet at Brahmasthan anchors room energy.'},
    'rug':          {'zones': ['Center', 'East', 'North'],
                     'rationale': 'Same as carpet.'},
    'blanket':      {'zones': ['South-West', 'South'],
                     'rationale': 'Blankets associated with rest — South-West.'},
    'pillow':       {'zones': ['South-West', 'South'],
                     'rationale': 'Pillows follow bed zone rules.'},
    'cushion':      {'zones': ['North', 'East', 'West'],
                     'rationale': 'Cushions follow sofa/chair zone rules.'},
    'bicycle':      {'zones': ['West', 'North-West'],
                     'rationale': 'Vehicles represent movement — North-West (Vayavya).'},
    'umbrella':     {'zones': ['West', 'North-West', 'East'],
                     'rationale': 'Shelter and rain (Jal) — North-West or near entrance.'},
    'backpack':     {'zones': ['West', 'North-West'],
                     'rationale': 'Travel bags belong in the zone of movement.'},
    'suitcase':     {'zones': ['West', 'North-West'],
                     'rationale': 'Luggage in North-West activates travel energy.'},
    'fan':          {'zones': ['West', 'North-West'],
                     'rationale': 'Air movement appliance — North-West (Vayavya/wind).'},
    'air conditioner': {'zones': ['West', 'North-West'],
                        'rationale': 'Air movement — North-West is the wind corner.'},
    'trash can':    {'zones': ['South', 'West', 'South-West'],
                     'rationale': 'Waste disposal belongs away from auspicious zones.'},
    'dustbin':      {'zones': ['South', 'West', 'South-West'],
                     'rationale': 'Same as trash can.'},
    'bin':          {'zones': ['South', 'West', 'South-West'],
                     'rationale': 'Same as trash can.'},
    'painting':     {'zones': ['North', 'East', 'North-East'],
                     'rationale': 'Art on North/East walls invites positive energy.'},
    'picture frame':{'zones': ['North', 'East', 'North-East'],
                     'rationale': 'Follows painting placement rules.'},
    'door':         {'zones': ['North', 'North-East', 'East'],
                     'rationale': 'Main door in North/East invites prosperity.'},
    'window':       {'zones': ['North', 'East', 'North-East'],
                     'rationale': 'Windows in North/East allow morning light and positive energy.'},
}

FURNITURE_ALIASES = {
    "wardrobe":     "cabinet",
    "almirah":      "cabinet",
    "tv unit":      "cabinet",
    "work desk":    "desk",
    "study table":  "desk",
    "couch":        "sofa",
    "dining table": "dining table",
    "double bed":   "bed",
    "single bed":   "bed",
    "dustbin":      "dustbin",
    "trash can":    "trash can",
}

ZONE_SPAWN_POINTS = {
    'South-West': (0.15, 0.75), 'South': (0.5, 0.8), 'West': (0.15, 0.5),
    'East': (0.85, 0.5), 'North': (0.5, 0.15), 'North-East': (0.85, 0.15),
    'North-West': (0.15, 0.15), 'South-East': (0.85, 0.75), 'Center': (0.5, 0.5)
}
VASTU_RULES = {obj: v['zones'] for obj, v in VASTU_RULES_FULL.items()}
ZONE_COLORS = {
    'North-East': '#d4edda', 'North': '#cce5ff', 'North-West': '#e2d9f3',
    'East': '#fff3cd', 'Center': '#f8f9fa', 'West': '#fde8d8',
    'South-East': '#ffe0b2', 'South': '#ffd7d7', 'South-West': '#e8d5c4',
}
ZONE_GRID = [
    ('North-West', 0,    0,    0.33, 0.33),
    ('North',      0.33, 0,    0.33, 0.33),
    ('North-East', 0.66, 0,    0.34, 0.33),
    ('West',       0,    0.33, 0.33, 0.34),
    ('Center',     0.33, 0.33, 0.34, 0.34),
    ('East',       0.66, 0.33, 0.34, 0.34),
    ('South-West', 0,    0.67, 0.33, 0.33),
    ('South',      0.33, 0.67, 0.34, 0.33),
    ('South-East', 0.66, 0.67, 0.34, 0.33),
]
ZONE_SHORT = {
    'North-East': 'NE', 'North-West': 'NW', 'South-East': 'SE', 'South-West': 'SW',
    'North': 'N', 'South': 'S', 'East': 'E', 'West': 'W', 'Center': 'C'
}
FURNITURE = {
    'sofa': (2.0, 0.8, 0.9), 'couch': (2.2, 0.8, 0.95), 'bed': (2.0, 0.6, 1.8),
    'tv': (1.2, 0.7, 0.15), 'television': (1.2, 0.7, 0.15),
    'chair': (0.5, 0.9, 0.5), 'stool': (0.4, 0.6, 0.4), 'bench': (1.4, 0.45, 0.5),
    'desk': (1.4, 0.75, 0.7), 'table': (1.4, 0.75, 0.8),
    'dining table': (1.6, 0.75, 1.0), 'nightstand': (0.5, 0.6, 0.4),
    'cabinet': (1.0, 1.8, 0.5), 'wardrobe': (1.2, 1.9, 0.6),
    'bookcase': (0.9, 1.8, 0.3), 'shelf': (0.8, 0.3, 0.2),
    'dresser': (1.0, 1.2, 0.5), 'cupboard': (1.0, 1.8, 0.5), 'drawer': (0.8, 0.7, 0.4),
    'refrigerator': (0.7, 1.8, 0.7), 'oven': (0.6, 0.9, 0.6),
    'microwave': (0.5, 0.35, 0.4), 'sink': (0.6, 0.2, 0.5),
    'toaster': (0.35, 0.22, 0.28), 'kettle': (0.22, 0.28, 0.22),
    'dishwasher': (0.6, 0.85, 0.6), 'washing machine': (0.6, 0.85, 0.6),
    'dryer': (0.6, 0.85, 0.6),
    'laptop': (0.35, 0.02, 0.25), 'computer': (0.2, 0.4, 0.2),
    'monitor': (0.5, 0.35, 0.1), 'keyboard': (0.4, 0.03, 0.15),
    'cell phone': (0.08, 0.01, 0.16), 'remote': (0.05, 0.02, 0.18),
    'toilet': (0.45, 0.75, 0.65), 'bathtub': (1.7, 0.55, 0.8),
    'shower': (0.9, 2.0, 0.9), 'towel': (0.6, 0.02, 0.3),
    'potted plant': (0.35, 0.7, 0.35), 'vase': (0.2, 0.4, 0.2),
    'clock': (0.3, 0.3, 0.1), 'mirror': (0.8, 1.2, 0.05),
    'book': (0.15, 0.05, 0.2), 'painting': (0.8, 0.6, 0.05),
    'picture frame': (0.5, 0.4, 0.04), 'candle': (0.06, 0.15, 0.06),
    'lamp': (0.25, 1.5, 0.25), 'chandelier': (0.6, 0.4, 0.6), 'light': (0.3, 0.3, 0.3),
    'curtain': (2.0, 2.4, 0.05), 'carpet': (2.0, 0.01, 3.0), 'rug': (1.5, 0.01, 2.0),
    'blanket': (1.5, 0.05, 2.0), 'pillow': (0.5, 0.15, 0.7), 'cushion': (0.4, 0.12, 0.4),
    'bicycle': (1.8, 1.0, 0.6), 'umbrella': (0.1, 1.0, 0.1),
    'backpack': (0.3, 0.5, 0.2), 'suitcase': (0.5, 0.7, 0.25), 'bag': (0.3, 0.4, 0.2),
    'fan': (0.4, 0.4, 0.15), 'air conditioner': (0.9, 0.3, 0.25),
    'trash can': (0.35, 0.5, 0.35), 'dustbin': (0.35, 0.5, 0.35), 'bin': (0.3, 0.4, 0.3),
    'door': (0.05, 2.1, 0.9), 'window': (0.1, 1.2, 1.0),
    'flower': (0.2, 0.3, 0.2),
}
FURN_COLORS = {
    'sofa': [100, 150, 200, 255], 'couch': [100, 150, 200, 255],
    'bed': [150, 100, 100, 255], 'tv': [30, 30, 30, 255], 'television': [30, 30, 30, 255],
    'chair': [139, 90, 60, 255], 'stool': [160, 120, 80, 255], 'bench': [160, 130, 90, 255],
    'desk': [180, 140, 100, 255], 'table': [170, 130, 90, 255],
    'dining table': [160, 120, 80, 255], 'nightstand': [150, 110, 70, 255],
    'cabinet': [140, 110, 80, 255], 'wardrobe': [130, 100, 70, 255],
    'bookcase': [160, 130, 100, 255], 'shelf': [150, 120, 90, 255],
    'dresser': [145, 115, 85, 255], 'cupboard': [140, 110, 80, 255],
    'drawer': [150, 120, 90, 255],
    'refrigerator': [220, 220, 220, 255], 'oven': [200, 200, 200, 255],
    'microwave': [180, 180, 180, 255], 'sink': [230, 230, 230, 255],
    'toaster': [190, 160, 130, 255], 'kettle': [200, 90, 50, 255],
    'dishwasher': [210, 210, 210, 255], 'washing machine': [220, 220, 220, 255],
    'dryer': [200, 200, 200, 255],
    'laptop': [60, 60, 60, 255], 'computer': [80, 80, 80, 255],
    'monitor': [50, 50, 50, 255], 'keyboard': [70, 70, 70, 255],
    'cell phone': [40, 40, 40, 255], 'remote': [60, 60, 60, 255],
    'toilet': [255, 255, 255, 255], 'bathtub': [240, 240, 255, 255],
    'shower': [200, 230, 255, 255], 'towel': [200, 180, 160, 255],
    'potted plant': [60, 140, 60, 255], 'vase': [200, 150, 100, 255],
    'clock': [50, 50, 50, 255], 'mirror': [180, 210, 230, 200],
    'book': [180, 140, 100, 255], 'painting': [210, 180, 140, 255],
    'picture frame': [160, 130, 100, 255], 'candle': [255, 220, 100, 255],
    'lamp': [240, 220, 100, 255], 'chandelier': [220, 190, 80, 255],
    'light': [255, 240, 150, 255],
    'curtain': [210, 180, 160, 200], 'carpet': [160, 110, 80, 200],
    'rug': [170, 120, 90, 200], 'blanket': [200, 180, 160, 255],
    'pillow': [220, 200, 180, 255], 'cushion': [190, 170, 150, 255],
    'bicycle': [80, 80, 200, 255], 'umbrella': [100, 180, 100, 255],
    'backpack': [100, 120, 180, 255], 'suitcase': [120, 100, 80, 255],
    'bag': [110, 90, 70, 255],
    'fan': [150, 190, 210, 255], 'air conditioner': [180, 210, 230, 255],
    'trash can': [100, 100, 100, 255], 'dustbin': [100, 100, 100, 255],
    'bin': [100, 100, 100, 255],
    'door': [180, 140, 100, 255], 'window': [180, 220, 240, 200],
    'flower': [220, 100, 130, 255],
}
ZONE_3D_COLORS = {
    'North-East': '#a8d8a8', 'North': '#aec6cf', 'North-West': '#c3b1e1',
    'East': '#ffd966', 'Center': '#eeeeee', 'West': '#f4a460',
    'South-East': '#ffb347', 'South': '#ff9999', 'South-West': '#c8a882',
}


# --------------------------
# LOAD HEAVY MODELS ONCE
# --------------------------
def load_shared_models():
    global YOLO_MODEL, MIDAS_MODEL, MIDAS_TRANSFORM
    if YOLO_MODEL is not None and MIDAS_MODEL is not None:
        return
    logger.info(f"Loading shared ML models on {DEVICE}")
    # Use YOLOv8m (medium) for better detection accuracy vs yolov8n (nano)
    try:
        YOLO_MODEL = YOLO('yolov8m.pt')
        logger.info("Loaded YOLOv8m")
    except Exception:
        YOLO_MODEL = YOLO('yolov8n.pt')
        logger.info("Fallback to YOLOv8n")
    try:
        MIDAS_MODEL = torch.hub.load('intel-isl/MiDaS', 'DPT_Large', trust_repo=True)
        MIDAS_TRANSFORM = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True)
    except Exception as e:
        logger.warning(f"Falling back to MiDaS_small: {str(e)}")
        MIDAS_MODEL = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', trust_repo=True)
        MIDAS_TRANSFORM = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True)
    MIDAS_MODEL.to(DEVICE).eval()


# --------------------------
# HELPER FUNCTIONS
# --------------------------
def upload_video():
    if __name__ != "__main__":
        raise RuntimeError("upload_video() only for local testing!")
    try:
        from tkinter import filedialog
        path = filedialog.askopenfilename()
        if not path:
            path = input("Enter full path to video: ").strip()
        return path
    except Exception:
        return input("Enter full path to video: ").strip()


def extract_frames_robust(video_path, sample_rate=5, max_frames=30):
    if not os.path.exists(video_path):
        for c in [video_path, './' + video_path, os.path.join(os.getcwd(), video_path)]:
            if os.path.exists(c):
                video_path = c
                break
        else:
            raise FileNotFoundError('Video not found: ' + video_path)
    print('Opening: ' + video_path)
    cap   = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError('Cannot open video.')
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total == 0 or W == 0 or H == 0:
        raise ValueError('Invalid video.')
    ar      = W / H
    is_pano = ar > 1.8
    video_info = dict(fps=float(fps), total_frames=total, width=W, height=H,
                      aspect_ratio=ar, duration=total / fps if fps > 0 else 0,
                      path=video_path, filename=os.path.basename(video_path),
                      is_panoramic=is_pano)
    print(f"{W}x{H} {round(fps,1)}fps {round(video_info['duration'],1)}s "
          f"{'[PANORAMIC]' if is_pano else '[NORMAL]'}")
    # Use lower effective sample rate so we get more frames per video
    eff_rate = max(1, sample_rate // 2) if is_pano else sample_rate
    frames, fid = [], 0
    while True:
        ret, frame = cap.read()
        if not ret or len(frames) >= max_frames:
            break
        if fid % eff_rate == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if is_pano:
                cw = int(H * 1.5)
                if cw < W:
                    sx = (W - cw) // 2
                    rgb = rgb[:, sx:sx + cw]
            frames.append(rgb)
        fid += 1
    cap.release()
    if not frames:
        raise ValueError('No frames extracted.')
    print(f'Extracted {len(frames)} frames (every {eff_rate})')
    return frames, video_info


def detect_objects_robust(frames, model, target_objects, confidence=0.15):
    """
    Detect objects across all frames.
    confidence=0.15 (lowered from 0.3) to maximise detections.
    Falls back to 0.10 if very few objects found.
    """
    all_dets = []
    print(f'Detecting in {len(frames)} frames (conf≥{confidence})...')
    for idx, frame in enumerate(frames):
        if (idx + 1) % 5 == 0:
            print(f'  {idx+1}/{len(frames)}...', end='\r')
        try:
            for r in model(frame, conf=confidence, verbose=False):
                for box in r.boxes:
                    lbl = model.names[int(box.cls[0])]
                    if lbl not in target_objects:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if x2 > x1 and y2 > y1:
                        all_dets.append(dict(
                            frame_id=idx, label=lbl,
                            bbox=[x1, y1, x2, y2],
                            confidence=float(box.conf[0])))
        except Exception as e:
            print(f'  warn frame {idx}: {e}')

    # If very few detections, retry with even lower confidence
    if len(set(d['label'] for d in all_dets)) < 2:
        print(f'  Only {len(all_dets)} detections — retrying with conf=0.10')
        for idx, frame in enumerate(frames):
            try:
                for r in model(frame, conf=0.10, verbose=False):
                    for box in r.boxes:
                        lbl = model.names[int(box.cls[0])]
                        if lbl not in target_objects:
                            continue
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        if x2 > x1 and y2 > y1:
                            existing = any(
                                d['frame_id'] == idx and d['label'] == lbl
                                for d in all_dets)
                            if not existing:
                                all_dets.append(dict(
                                    frame_id=idx, label=lbl,
                                    bbox=[x1, y1, x2, y2],
                                    confidence=float(box.conf[0])))
            except Exception:
                pass

    print(f'\nDetected {len(all_dets)} instances of '
          f'{len(set(d["label"] for d in all_dets))} unique objects')
    for obj, n in Counter(d['label'] for d in all_dets).most_common():
        print(f'  {obj.ljust(22)}: {n}')
    return all_dets


def estimate_depth(image, model, tfm, dev):
    try:
        with torch.no_grad():
            pred = model(tfm.dpt_transform(image).to(dev))
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=image.shape[:2],
                mode='bicubic', align_corners=False).squeeze()
        dm = pred.cpu().numpy()
        return np.nan_to_num(dm, nan=0., posinf=0., neginf=0.)
    except Exception as e:
        print('  depth error: ' + str(e))
        return np.ones(image.shape[:2]) * 0.5


def get_vastu_zone(cx, cy):
    v = 'North' if cy < 0.33 else ('Center' if cy < 0.66 else 'South')
    h = 'West'  if cx < 0.33 else ('Center' if cx < 0.66 else 'East')
    if v == 'Center' and h == 'Center':
        return 'Center'
    if v == 'Center':
        return h
    if h == 'Center':
        return v
    return v + '-' + h


def coords_to_zone(cx, cy):
    return get_vastu_zone(cx, cy)


def check_compliance(objects, rules):
    results, viol = [], 0
    for obj, d in objects.items():
        zone    = d['zone']
        allowed = rules.get(obj, [])
        ok      = zone in allowed if allowed else True
        if not ok:
            viol += 1
        results.append(dict(object=obj, zone=zone, allowed=allowed, compliant=ok))
    return results, viol


def init_population(objects, pop_size=50):
    return [{obj: (random.random(), random.random()) for obj in objects}
            for _ in range(pop_size)]


def mutate(ind, rate=0.3, sigma=0.15):
    new = ind.copy()
    for obj in new:
        if random.random() < rate:
            cx, cy = new[obj]
            new[obj] = (float(np.clip(cx + random.gauss(0, sigma), 0, 1)),
                        float(np.clip(cy + random.gauss(0, sigma), 0, 1)))
    return new


def crossover(p1, p2):
    return {obj: p1[obj] if random.random() < 0.5 else p2[obj] for obj in p1}


def calculate_fitness(layout, original, rules):
    vp = mp = op = 0
    for obj, (cx, cy) in layout.items():
        orig_data = original[obj]
        weight = 3 if orig_data.get('is_essential', False) else 1
        if rules.get(obj) and coords_to_zone(cx, cy) not in rules[obj]:
            vp += 10 * weight
        mp += 0.2 * weight * np.sqrt(
            (cx - orig_data['cx']) ** 2 + (cy - orig_data['cy']) ** 2)
    pos = list(layout.items())
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            dist = np.sqrt((pos[i][1][0] - pos[j][1][0]) ** 2 +
                           (pos[i][1][1] - pos[j][1][1]) ** 2)
            if dist < 0.15:
                i_ess = original[pos[i][0]].get('is_essential', False)
                j_ess = original[pos[j][0]].get('is_essential', False)
                op += 5 * (3 if (i_ess and j_ess) else 1)
    return vp + mp + op


def run_ga(orig, rules, generations=100, pop_size=50):
    print(f'GA: {generations} gen x {pop_size} pop...')
    pop  = init_population(orig.keys(), pop_size)
    hist = []
    for gen in range(generations):
        scored = sorted([(ind, calculate_fitness(ind, orig, rules)) for ind in pop],
                        key=lambda x: x[1])
        pop  = [ind for ind, _ in scored]
        hist.append(scored[0][1])
        if (gen + 1) % 20 == 0:
            print(f'  Gen {gen+1}: fitness={round(hist[-1],3)}')
        parents = pop[:max(1, int(0.3 * pop_size))]
        new_pop = parents[:]
        while len(new_pop) < pop_size:
            new_pop.append(mutate(crossover(
                random.choice(parents), random.choice(parents))))
        pop = new_pop
    final      = [(ind, calculate_fitness(ind, orig, rules)) for ind in pop]
    best, score = min(final, key=lambda x: x[1])
    print(f'GA done | {round(hist[0],3)} -> {round(score,3)} '
          f'(delta {round(hist[0]-score,3)})')
    return best, hist


def estimate_camera_params(W, H, fov=60, is_panoramic=False):
    if is_panoramic:
        fov = 120
    fx = W / (2 * np.tan(np.radians(fov) / 2))
    return dict(fx=float(fx), fy=float(fx), cx=float(W/2), cy=float(H/2),
                width=W, height=H, fov=fov, is_panoramic=is_panoramic)


def get_depth_at(cx_px, cy_px, dm, margin=15):
    H, W = dm.shape
    y1, y2 = max(0, int(cy_px-margin)), min(H, int(cy_px+margin))
    x1, x2 = max(0, int(cx_px-margin)), min(W, int(cx_px+margin))
    return np.median(dm[y1:y2, x1:x2]) if y2>y1 and x2>x1 else dm[int(cy_px), int(cx_px)]


def pixel_to_3d(cx_px, cy_px, dv, cam, depth_map):
    di = 1.0 / (dv + 1e-6)
    dn = (di - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
    Z  = 2.0 + dn * 6.0
    return (float((cx_px - cam['cx']) * Z / cam['fx']),
            float((cy_px - cam['cy']) * Z / cam['fy']),
            float(Z))


def create_room_mesh(dm, floor_level=0, ceiling_height=2.5):
    di   = 1.0 / (dm + 1e-6)
    d_c  = np.percentile(di.flatten(), 10)
    d_f  = np.percentile(di.flatten(), 90)
    rdep = 3.0 + (d_f - d_c) * 2.0
    rwid = rdep * 1.3
    xmin, xmax = -rwid/2, rwid/2
    verts = np.array([
        [xmin, floor_level, 0],    [xmax, floor_level, 0],
        [xmax, floor_level, rdep], [xmin, floor_level, rdep],
        [xmin, ceiling_height, 0], [xmax, ceiling_height, 0],
        [xmax, ceiling_height, rdep], [xmin, ceiling_height, rdep],
    ])
    faces = np.array([
        [0,2,1],[0,3,2],[4,5,6],[4,6,7],
        [0,1,5],[0,5,4],[1,2,6],[1,6,5],
        [2,3,7],[2,7,6],[3,0,4],[3,4,7],
    ])
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.visual.face_colors = [210, 210, 215, 100]
    print(f'Room: {round(rwid,2)}m W x {round(rdep,2)}m D x {ceiling_height}m H')
    return mesh, (rwid, rdep, ceiling_height)


def create_furniture_mesh(name, pos3d):
    size = FURNITURE.get(name, (0.8, 0.8, 0.8))
    box  = trimesh.creation.box(extents=size)
    X, Y, Z = pos3d
    box.apply_translation([X, size[1]/2, Z])
    box.visual.face_colors = FURN_COLORS.get(name, [160, 160, 160, 255])
    return box


def create_scene(room_mesh, objs):
    scene = trimesh.Scene()
    scene.add_geometry(room_mesh, node_name='room', geom_name='room')
    for name, d in objs.items():
        scene.add_geometry(create_furniture_mesh(name, d['position_3d']),
                           node_name=name, geom_name=name)
    return scene


def draw_vastu_grid(ax, title):
    for zone, x, y, w, h in ZONE_GRID:
        ax.add_patch(plt.Rectangle((x,y), w, h,
                                   facecolor=ZONE_COLORS.get(zone,'#ffffff'),
                                   edgecolor='#aaaaaa', linewidth=1.2, zorder=1))
        ax.text(x+w/2, y+h/2, ZONE_SHORT.get(zone,zone),
                ha='center', va='center', fontsize=11, color='#555555', zorder=2,
                bbox=dict(boxstyle='round,pad=0.3',fc='white',alpha=0.5,ec='none'))
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect('equal')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=14)
    ax.set_xlabel('<- West                         East ->', fontsize=10)
    ax.set_ylabel('<- North                       South ->', fontsize=10)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


def visualize_3d_scene(scene, title, objs_dict, save_path):
    fig = plt.figure(figsize=(15,10))
    ax  = fig.add_subplot(111, projection='3d')
    for name, mesh in scene.geometry.items():
        if not len(mesh.vertices):
            continue
        fv = mesh.vertices[mesh.faces]
        if name == 'room':
            color = [0.88,0.88,0.90,0.22]
        else:
            try:
                c = mesh.visual.face_colors[0][:4]/255.0
                color = [float(c[0]),float(c[1]),float(c[2]),0.80]
            except:
                color = [0.7,0.7,0.7,0.80]
        poly = Poly3DCollection(fv, facecolor=color[:3], edgecolor='#555555',
                                linewidth=0.25, alpha=color[3])
        ax.add_collection3d(poly)
    all_v = np.vstack([m.vertices for m in scene.geometry.values() if len(m.vertices)])
    pad = 0.6
    ax.set_xlim(all_v[:,0].min()-pad, all_v[:,0].max()+pad)
    ax.set_ylim(all_v[:,1].min()-pad, all_v[:,1].max()+pad)
    ax.set_zlim(all_v[:,2].min()-pad, all_v[:,2].max()+pad)
    ax.set_xlabel('X (m)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
    ax.set_zlabel('Z (m)', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.view_init(elev=28, azim=40)
    patches = [mpatches.Patch(color=c, label=z) for z,c in ZONE_3D_COLORS.items()]
    ax.legend(handles=patches, loc='upper left', fontsize=7.5, ncol=2,
              title='Vastu Zone', title_fontsize=8)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def compute_vastu_score(name, stabilized_obj, new_zone, dist, detections, internal_name):
    MAX_DIST    = 1.5
    allowed_zones = VASTU_RULES.get(internal_name, [])
    zone_score  = 50 if (new_zone in allowed_zones or not allowed_zones) else 0
    det_confs   = [d['confidence'] for d in detections if d['label'] == name]
    conf_score  = round(float(np.mean(det_confs)) * 30, 1) if det_confs else 15.0
    move_score  = round(max(0.0, 20.0*(1 - min(dist, MAX_DIST)/MAX_DIST)), 1)
    total       = round(zone_score + conf_score + move_score, 1)
    label       = ('Excellent' if total >= 85 else 'Good' if total >= 65
                   else 'Fair' if total >= 45 else 'Poor')
    breakdown   = {'zone_compliance': zone_score,
                   'detection_confidence': conf_score,
                   'movement_stability': move_score}
    return total, label, breakdown


# --------------------------
# MAIN PIPELINE
# --------------------------
def run_full_vastu_pipeline(
    user_video_path: str,
    user_room_type: str,
    user_selected_furniture: list = None,
    base_output_root: str = "media/vastu_results"
) -> dict:
    print("✅ Video Path:", user_video_path)
    print("✅ Room Type:",  user_room_type)
    print("✅ Furniture:", user_selected_furniture)

    load_shared_models()
    user_selected_furniture = user_selected_furniture or []

    run_id     = str(uuid.uuid4())
    output_dir = os.path.join(base_output_root, run_id)
    for d in [output_dir,
              output_dir+'/frames', output_dir+'/3d_models',
              output_dir+'/renders', output_dir+'/visualizations',
              output_dir+'/data']:
        os.makedirs(d, exist_ok=True)
    logger.info(f"Starting Vastu run {run_id} for {user_room_type}")

    try:
        # Step 1: Frame extraction (max 30 frames, sample every 5th)
        frames, video_info = extract_frames_robust(
            user_video_path, FRAME_SAMPLE_RATE, max_frames=30)
        video_info['room_type'] = user_room_type

        total_frames   = len(frames)
        sample_indices = list(dict.fromkeys([0, total_frames//2, -1]))
        fig, axes      = plt.subplots(1, len(sample_indices), figsize=(16,5))
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        fig.suptitle('Sample Frames: '+video_info['filename'],
                     fontsize=14, fontweight='bold')
        for ax, idx in zip(axes, sample_indices):
            frame_number = idx if idx >= 0 else total_frames+idx
            ax.imshow(frames[idx])
            ax.set_title(f'Frame {frame_number}')
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_dir+'/visualizations/01_sample_frames.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Step 2: YOLO detection (confidence=0.15 for max detections)
        detections = detect_objects_robust(
            frames, YOLO_MODEL, ROOM_OBJECTS, confidence=0.15)

        if detections:
            H0, W0, _ = frames[0].shape
            heatmap    = np.zeros((H0,W0), dtype=np.float32)
            for d in detections:
                x1,y1,x2,y2 = d['bbox']
                x1,y1,x2,y2 = max(0,x1),max(0,y1),min(W0,x2),min(H0,y2)
                heatmap[y1:y2, x1:x2] += d['confidence']
            heatmap /= (heatmap.max()+1e-6)
            mid        = len(frames)//2
            frame_dets = [d for d in detections if d['frame_id']==mid]
            fig,(ax1,ax2) = plt.subplots(1,2,figsize=(18,7))
            ax1.imshow(frames[mid].copy())
            for d in frame_dets:
                x1,y1,x2,y2 = d['bbox']
                ax1.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,
                                        fill=False,edgecolor='#ff4444',lw=2))
                ax1.text(x1,max(0,y1-6),
                         d['label']+' '+str(round(d['confidence'],2)),
                         color='white',backgroundcolor='#cc0000',fontsize=7.5)
            ax1.set_title(f'Detections (frame {mid})',fontsize=13,fontweight='bold')
            ax1.axis('off')
            ax2.imshow(frames[mid],alpha=0.45)
            im = ax2.imshow(heatmap,cmap='hot',alpha=0.65)
            plt.colorbar(im,ax=ax2,fraction=0.036,label='Detection density')
            ax2.set_title('Detection heatmap',fontsize=13,fontweight='bold')
            ax2.axis('off')
            plt.tight_layout()
            plt.savefig(output_dir+'/visualizations/02_detections_heatmap.png',
                        dpi=150,bbox_inches='tight')
            plt.close(fig)

        # Step 3: Depth estimation
        rep_frame = frames[len(frames)//2]
        depth_map = estimate_depth(rep_frame, MIDAS_MODEL, MIDAS_TRANSFORM, DEVICE)
        dn        = 1.0/(depth_map+1e-6)
        dn        = (dn-dn.min())/(dn.max()-dn.min())
        edges     = cv2.Canny(
            ((depth_map-depth_map.min())/(depth_map.max()-depth_map.min())*255
             ).astype(np.uint8),50,150)
        fig,axes = plt.subplots(1,5,figsize=(26,5))
        fig.suptitle('Depth Analysis',fontsize=14,fontweight='bold')
        for ax,img,ttl,cmap in zip(axes,
            [rep_frame,depth_map,dn,1-dn,edges],
            ['Original','Raw depth','Normalised','Inverted','Depth edges'],
            [None,'plasma','viridis','inferno','gray']):
            im = ax.imshow(img,cmap=cmap); ax.set_title(ttl,fontsize=10); ax.axis('off')
            if cmap: plt.colorbar(im,ax=ax,fraction=0.04,pad=0.02)
        plt.tight_layout()
        plt.savefig(output_dir+'/visualizations/03_depth_analysis.png',
                    dpi=150,bbox_inches='tight')
        plt.close(fig)

        # Step 4: Stabilise objects + add user-selected essential furniture
        IMG_H, IMG_W, _ = frames[0].shape
        for obj in detections:
            x1,y1,x2,y2 = obj['bbox']
            cx,cy = (x1+x2)/2, (y1+y2)/2
            obj['cx_norm']   = float(np.clip(cx/IMG_W, 0, 1))
            obj['cy_norm']   = float(np.clip(cy/IMG_H, 0, 1))
            obj['area_norm'] = float((x2-x1)*(y2-y1)/(IMG_W*IMG_H))
            obj['cx_pixel']  = float(cx)
            obj['cy_pixel']  = float(cy)

        object_detections = defaultdict(list)
        for obj in detections:
            object_detections[obj['label']].append(
                (obj['cx_norm'],obj['cy_norm'],obj['area_norm'],
                 obj['cx_pixel'],obj['cy_pixel']))

        stabilized_objects = {}
        for label, dl in object_detections.items():
            top = sorted(dl,key=lambda x:x[2],reverse=True)[:10]  # top-10
            stabilized_objects[label] = dict(
                cx=float(np.mean([d[0] for d in top])),
                cy=float(np.mean([d[1] for d in top])),
                area=float(np.mean([d[2] for d in top])),
                cx_pixel=float(np.mean([d[3] for d in top])),
                cy_pixel=float(np.mean([d[4] for d in top])),
                is_essential=False)

        if user_selected_furniture:
            for essential_obj in user_selected_furniture:
                if essential_obj in stabilized_objects:
                    stabilized_objects[essential_obj]['is_essential'] = True
                else:
                    logger.info(f"Adding user-selected (missed by YOLO): {essential_obj}")
                    internal_name    = FURNITURE_ALIASES.get(essential_obj, essential_obj)
                    allowed_zones    = VASTU_RULES.get(internal_name, ["South-West"])
                    first_valid_zone = allowed_zones[0]
                    default_cx, default_cy = ZONE_SPAWN_POINTS.get(
                        first_valid_zone, (0.5, 0.5))
                    default_size = FURNITURE.get(internal_name, (0.8, 0.8, 0.8))
                    stabilized_objects[essential_obj] = dict(
                        cx=default_cx + random.uniform(-0.05, 0.05),
                        cy=default_cy + random.uniform(-0.05, 0.05),
                        area=(default_size[0]*default_size[2])/(IMG_W*IMG_H)*1e6,
                        cx_pixel=default_cx*IMG_W + random.randint(-20,20),
                        cy_pixel=default_cy*IMG_H + random.randint(-20,20),
                        is_essential=True,
                        internal_name=internal_name)

        print(f"\nFinal object list ({len(stabilized_objects)}):")
        for name, d in stabilized_objects.items():
            print(f"  {name.ljust(22)} ess={d.get('is_essential',False)}")

        # Step 5: 3D coordinate calculation
        camera_params = estimate_camera_params(
            IMG_W, IMG_H, CAMERA_FOV, is_panoramic=video_info['is_panoramic'])
        objects_3d = {}
        for name, d in stabilized_objects.items():
            dv      = get_depth_at(d['cx_pixel'], d['cy_pixel'], depth_map)
            X,Y,Z   = pixel_to_3d(d['cx_pixel'], d['cy_pixel'], dv,
                                   camera_params, depth_map)
            objects_3d[name] = dict(position_3d=(X,Y,Z),
                                    position_2d=(d['cx'],d['cy']),
                                    depth_value=float(dv))
            d['zone'] = get_vastu_zone(d['cx'], d['cy'])
            objects_3d[name]['zone'] = d['zone']

        # Step 6: Initial compliance
        compliance, violations = check_compliance(stabilized_objects, VASTU_RULES)
        if len(compliance) >= 3:
            objs_list   = [c['object'] for c in compliance]
            compliant_v = [1 if c['compliant'] else 0 for c in compliance]
            angles      = np.linspace(0,2*np.pi,len(objs_list),endpoint=False).tolist()
            vals        = compliant_v + [compliant_v[0]]
            ang         = angles + angles[:1]
            fig,ax = plt.subplots(subplot_kw=dict(polar=True),figsize=(7,7))
            ax.plot(ang,vals,'o-',linewidth=2,color='#2c7bb6')
            ax.fill(ang,vals,alpha=0.25,color='#2c7bb6')
            ax.set_xticks(angles); ax.set_xticklabels(objs_list,size=9)
            ax.set_yticks([0,1]); ax.set_yticklabels(['Violation','Compliant'],size=8)
            ax.set_title('Vastu Compliance Radar',fontsize=14,fontweight='bold',pad=20)
            plt.tight_layout()
            plt.savefig(output_dir+'/visualizations/04_compliance_radar.png',
                        dpi=150,bbox_inches='tight')
            plt.close(fig)
        n_obj = max(len(stabilized_objects),1)
        initial_compliance = round((1-violations/n_obj)*100,1)

        # Step 7: GA optimisation
        optimized_layout, fitness_history = run_ga(
            stabilized_objects, VASTU_RULES, GA_GENERATIONS, GA_POPULATION)

        fig,ax = plt.subplots(figsize=(13,6))
        ax.plot(fitness_history,linewidth=2.5,color='#2c5aa0',label='Best fitness')
        ax.fill_between(range(len(fitness_history)),fitness_history,
                        alpha=0.12,color='#2c5aa0')
        improvement = fitness_history[0]-fitness_history[-1]
        ax.annotate(f'Delta {round(improvement,2)}',
                    xy=(len(fitness_history)-1,fitness_history[-1]),
                    xytext=(len(fitness_history)*0.65,fitness_history[0]*0.65),
                    arrowprops=dict(arrowstyle='->',color='#cc3333'),
                    fontsize=11,color='#cc3333',fontweight='bold')
        ax.set_xlabel('Generation',fontsize=13,fontweight='bold')
        ax.set_ylabel('Best Fitness (lower=better)',fontsize=13,fontweight='bold')
        ax.set_title('Genetic Algorithm Convergence',fontsize=15,fontweight='bold')
        ax.grid(True,alpha=0.25,linestyle='--'); ax.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(output_dir+'/visualizations/05_ga_convergence.png',
                    dpi=150,bbox_inches='tight')
        plt.close(fig)

        # Step 8: Optimised compliance
        optimized_objects_3d = {}
        for name,(cx,cy) in optimized_layout.items():
            px,py   = cx*IMG_W, cy*IMG_H
            dv      = get_depth_at(px,py,depth_map)
            X,Y,Z   = pixel_to_3d(px,py,dv,camera_params,depth_map)
            optimized_objects_3d[name] = dict(
                position_3d=(X,Y,Z), position_2d=(cx,cy),
                zone=coords_to_zone(cx,cy), depth_value=float(dv))

        opt_check = {o:{'zone':d['zone']} for o,d in optimized_objects_3d.items()}
        opt_compliance, opt_violations = check_compliance(opt_check, VASTU_RULES)
        final_compliance = round((1-opt_violations/max(len(opt_check),1))*100,1)

        # Step 9: 2D layout comparison
        fig,(ax1,ax2) = plt.subplots(1,2,figsize=(22,10))
        draw_vastu_grid(ax1,'Current Layout: '+video_info['filename'])
        draw_vastu_grid(ax2,'Optimised Layout (Vastu-Compliant)')
        for obj,d in stabilized_objects.items():
            ok     = d['zone'] in VASTU_RULES.get(obj,[d['zone']])
            color  = '#22aa44' if ok else '#dd3333'
            marker = 'o' if ok else 'X'
            ax1.scatter(d['cx'],d['cy'],s=480,color=color,alpha=0.85,
                        edgecolors='white',linewidths=1.8,marker=marker,zorder=5)
            ax1.text(d['cx'],d['cy']-0.055,obj,ha='center',fontsize=8.5,zorder=6,
                     bbox=dict(boxstyle='round,pad=0.25',fc='white',
                               alpha=0.88,ec=color,lw=1))
        handles = [mpatches.Patch(color='#22aa44',label='Compliant'),
                   mpatches.Patch(color='#dd3333',label='Violation')]
        ax1.legend(handles=handles,loc='lower right',fontsize=10)
        for obj,(cx,cy) in optimized_layout.items():
            ax2.scatter(cx,cy,s=480,color='#22aa44',alpha=0.85,
                        edgecolors='white',linewidths=1.8,zorder=5)
            ax2.text(cx,cy-0.055,obj,ha='center',fontsize=8.5,zorder=6,
                     bbox=dict(boxstyle='round,pad=0.25',fc='#e8f5e9',
                               alpha=0.9,ec='#22aa44',lw=1))
            ocx,ocy = stabilized_objects[obj]['cx'],stabilized_objects[obj]['cy']
            if abs(cx-ocx)>0.04 or abs(ocy-cy)>0.04:
                ax2.add_patch(FancyArrowPatch((ocx,ocy),(cx,cy),
                              arrowstyle='->',mutation_scale=18,
                              lw=2,color='#1565c0',alpha=0.55,zorder=4))
        plt.tight_layout()
        plt.savefig(output_dir+'/visualizations/06_layout_comparison_2d.png',
                    dpi=200,bbox_inches='tight')
        plt.close(fig)

        # Step 10: 3D mesh
        room_mesh, room_dimensions = create_room_mesh(depth_map)
        current_scene   = create_scene(room_mesh, objects_3d)
        optimized_scene = create_scene(room_mesh, optimized_objects_3d)
        out3d = output_dir+'/3d_models'
        for label,scene in [('current',current_scene),('optimised',optimized_scene)]:
            for ext in ('obj','glb'):
                try: scene.export(out3d+'/'+label+'_layout.'+ext)
                except Exception as e: logger.warning(f"Export {label}.{ext}: {e}")
        visualize_3d_scene(current_scene,'Current Layout (3D)',objects_3d,
                           output_dir+'/renders/current_3d_view.png')
        visualize_3d_scene(optimized_scene,'Optimised Layout (3D)',optimized_objects_3d,
                           output_dir+'/renders/optimized_3d_view.png')

        # Step 11: Build recommendations
        recommendations = []
        changes_needed  = 0
        for name in stabilized_objects:
            old_zone = stabilized_objects[name]['zone']
            new_zone = optimized_objects_3d[name]['zone']
            old_pos  = objects_3d[name]['position_3d']
            new_pos  = optimized_objects_3d[name]['position_3d']
            dist     = float(np.linalg.norm(np.array(new_pos)-np.array(old_pos)))
            needs    = old_zone != new_zone
            if needs: changes_needed += 1
            internal_name = stabilized_objects[name].get('internal_name', name)
            rationale     = VASTU_RULES_FULL.get(
                internal_name,{}).get('rationale','General Vastu guidance.')
            vastu_score,score_label,score_breakdown = compute_vastu_score(
                name,stabilized_objects[name],new_zone,dist,detections,internal_name)
            recommendations.append(dict(
                object=name,
                is_essential=stabilized_objects[name].get('is_essential',False),
                current_zone=old_zone,
                recommended_zone=new_zone,
                current_position_3d=[float(x) for x in old_pos],
                recommended_position_3d=[float(x) for x in new_pos],
                movement_distance_m=dist,
                action_needed=needs,
                vastu_rationale=rationale,
                vastu_score=vastu_score,
                vastu_score_label=score_label,
                score_breakdown=score_breakdown,
            ))
        recommendations.sort(key=lambda x:(-x['is_essential'],x['action_needed']))

        # Step 12: Export JSON
        results = {
            'run_id': run_id,
            'video_info': {k:video_info[k] for k in
                           ('filename','path','room_type','width','height',
                            'fps','duration','is_panoramic')},
            'camera_params': {k:v for k,v in camera_params.items()
                              if isinstance(v,(int,float,bool))},
            'room_dimensions': dict(width_m=float(room_dimensions[0]),
                                    depth_m=float(room_dimensions[1]),
                                    height_m=float(room_dimensions[2])),
            'detected_objects': len(stabilized_objects),
            'user_selected_furniture': user_selected_furniture,
            'initial_violations': int(violations),
            'optimized_violations': int(opt_violations),
            'initial_compliance_pct': initial_compliance,
            'final_compliance_pct': final_compliance,
            'improvement_pct': float((violations-opt_violations)/max(violations,1)*100)
                               if violations>0 else 0,
            'recommendations': recommendations,
            'ga_stats': dict(generations=GA_GENERATIONS, population_size=GA_POPULATION,
                             initial_fitness=float(fitness_history[0]),
                             final_fitness=float(fitness_history[-1]),
                             improvement=float(fitness_history[0]-fitness_history[-1])),
            'public_urls': {
                'html_report':       f'/media/vastu_results/{run_id}/data/vastu_report.html',
                'current_3d_glb':    f'/media/vastu_results/{run_id}/3d_models/current_layout.glb',
                'optimized_3d_glb':  f'/media/vastu_results/{run_id}/3d_models/optimised_layout.glb',
                'compliance_radar':  f'/media/vastu_results/{run_id}/visualizations/04_compliance_radar.png',
                'layout_comparison': f'/media/vastu_results/{run_id}/visualizations/06_layout_comparison_2d.png',
            }
        }
        with open(output_dir+'/data/complete_results.json','w') as f:
            json.dump(results,f,indent=2)

        # Step 13: HTML report (unchanged structure)
        def score_pill_html(score,label,breakdown):
            if score>=85:   bg,fg='#e8f5e9','#2e7d32'
            elif score>=65: bg,fg='#fff8e1','#f57f17'
            elif score>=45: bg,fg='#fff3e0','#e65100'
            else:           bg,fg='#ffebee','#b71c1c'
            return (f'<div style="background:{bg};color:{fg};border-radius:6px;'
                    f'padding:5px 8px;font-weight:700;font-size:15px;text-align:center">'
                    f'{score}/100</div>'
                    f'<div style="font-size:11px;color:{fg};text-align:center;margin-top:3px;'
                    f'font-weight:600">{label}</div>'
                    f'<div style="font-size:10px;color:#777;margin-top:5px;line-height:1.7">'
                    f'&#x2713; Zone: <b>{breakdown["zone_compliance"]}</b>/50<br>'
                    f'&#x2713; Conf: <b>{breakdown["detection_confidence"]}</b>/30<br>'
                    f'&#x2713; Stability: <b>{breakdown["movement_stability"]}</b>/20'
                    f'</div>')

        rows = ''
        for r in recommendations:
            dist_s = str(round(r['movement_distance_m'],2))+' m' if r['action_needed'] else '-'
            rat    = r.get('vastu_rationale','')[:150]+('...' if len(r.get('vastu_rationale',''))>150 else '')
            clr    = '#d32f2f' if r['action_needed'] else '#388e3c'
            ess_badge = (' <span style="background:#1565c0;color:#fff;border-radius:4px;'
                         'padding:1px 5px;font-size:10px;vertical-align:middle">Essential</span>'
                         if r['is_essential'] else '')
            rows += ('<tr><td><strong>'+r['object']+'</strong>'+ess_badge+'</td>'
                     '<td>'+r['current_zone']+'</td><td>'+r['recommended_zone']+'</td>'
                     '<td>'+dist_s+'</td>'
                     '<td style="color:'+clr+';font-weight:600">'+
                     ('MOVE' if r['action_needed'] else 'OK')+'</td>'
                     '<td>'+score_pill_html(r['vastu_score'],r['vastu_score_label'],
                                            r['score_breakdown'])+'</td>'
                     '<td style="font-size:11px;color:#555">'+rat+'</td></tr>')

        html = ('<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
                '<title>Vastu 3D Report</title>'
                '<style>body{font-family:system-ui,sans-serif;max-width:1300px;margin:0 auto;'
                'padding:24px;background:#f5f5f5;color:#222}'
                'h1{color:#2c5aa0;border-bottom:2px solid #2c5aa0;padding-bottom:8px}'
                'table{width:100%;border-collapse:collapse;background:#fff;border-radius:10px;'
                'overflow:hidden;box-shadow:0 1px 4px #0002;margin-top:16px}'
                'th{background:#2c5aa0;color:#fff;padding:10px 12px;text-align:left;font-size:13px}'
                'td{padding:9px 12px;border-bottom:1px solid #eee;font-size:13px;vertical-align:top}'
                'tr:hover td{background:#f0f4ff}</style></head><body>'
                '<h1>Vastu 3D Analysis Report</h1>'
                '<p>Video: <strong>'+video_info['filename']+'</strong> | Room: '+user_room_type+'</p>'
                '<table><thead><tr><th>Object</th><th>Current Zone</th>'
                '<th>Recommended Zone</th><th>Distance</th>'
                '<th>Status</th><th>Vastu Score</th><th>Rationale</th>'
                '</tr></thead><tbody>'+rows+'</tbody></table>'
                '<div style="margin-top:32px;font-size:11px;color:#999;text-align:center">'
                'Vastu 3D Pipeline v2.2 | YOLOv8 + MiDaS + GA</div></body></html>')
        with open(output_dir+'/data/vastu_report.html','w') as f:
            f.write(html)

        # Step 14: Model pickle
        model_data = {
            'version': '2.2',
            'created_at': datetime.datetime.now().isoformat(),
            'video_filename': video_info.get('filename', user_video_path),
            'vastu_rules': VASTU_RULES,
            'ga_config': dict(generations=GA_GENERATIONS, population_size=GA_POPULATION,
                              mutation_rate=0.3, mutation_sigma=0.15, elite_ratio=0.3),
            'training_history': dict(
                fitness_history=[float(f) for f in fitness_history],
                initial_fitness=float(fitness_history[0]),
                final_fitness=float(fitness_history[-1])),
            'best_layout': {obj:{'cx':float(p[0]),'cy':float(p[1])}
                            for obj,p in optimized_layout.items()},
            'statistics': dict(initial_violations=int(violations),
                               final_violations=int(opt_violations),
                               objects_analyzed=len(stabilized_objects),
                               changes_needed=changes_needed),
        }
        with open(output_dir+'/data/vastu_ga_model.pkl','wb') as f: pickle.dump(model_data,f)
        with open(output_dir+'/data/vastu_ga_model.json','w') as f: json.dump(model_data,f,indent=2)

        logger.info(f"Run {run_id} done. Compliance: {final_compliance}%")
        return results

    except Exception as e:
        logger.error(f"Run {run_id} failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    local_video              = r"C:\Users\Lenovo\Downloads\room.mp4"
    local_room_type          = "bedroom"
    local_selected_furniture = ["bed", "wardrobe"]
    test_results = run_full_vastu_pipeline(
        user_video_path=local_video,
        user_room_type=local_room_type,
        user_selected_furniture=local_selected_furniture)
    print("\n✅ Local test complete!")
    print(json.dumps(test_results, indent=2))