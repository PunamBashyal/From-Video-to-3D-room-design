# back/views.py
# ─────────────────────────────────────────────────────────────────────────────
# Fixes:
#  1. furniture_data from frontend is {"beds":1,"wardrobe":1} →
#     correctly convert to ["bed","wardrobe"] list for pipeline
#  2. 3D render: detected-only in current, full in optimised
#  3. Direction index image: every item + zone + compass direction
#  4. FIX: _generate_renders() is now actually CALLED in post()
#  5. FIX: run_id None guard added
#  6. FIX: Full traceback logging
#  7. NEW: Interactive Three.js HTML — emoji icons on boxes, compass arrows,
#          hover tooltips, orbiting camera, toggle current/optimised
#  8. FIX: Ghost logic — essential+detected furniture shows as solid teal box.
#          Ghost ONLY when essential AND not_detected_in_video=True.
#  9. FIX: ALL detected furniture from video now shown in current layout.
#          _build_current_list() merges raw detections + recommendations so
#          purely-detected items that were dropped from recommendations are
#          still rendered. Also tries every key the pipeline might use:
#          'detections', 'detected_objects', 'current_layout',
#          'detected_furniture', falling back to 'recommendations' only.
# ─────────────────────────────────────────────────────────────────────────────

import os, json, uuid, logging, traceback
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from rest_framework.views    import APIView
from rest_framework.response import Response
from rest_framework          import status
from django.conf             import settings

from back.ml_models.final_vastu.final.vastu_3d_full import (
    run_full_vastu_pipeline,
    load_shared_models,
    extract_frames_robust,
    estimate_depth,
    DEVICE, MIDAS_MODEL, MIDAS_TRANSFORM,
    FRAME_SAMPLE_RATE, FURNITURE, FURN_COLORS,
)

logger        = logging.getLogger(__name__)
BASE_DIR      = settings.BASE_DIR
MEDIA_ROOT    = getattr(settings, 'MEDIA_ROOT', os.path.join(BASE_DIR, 'media'))
ML_MODELS_DIR = os.path.join(BASE_DIR, 'back', 'ml_models', 'MajorProject')

_UNET = None

# ─────────────────────────────────────────────────────────────────────────────
# Zone / Furniture constants (must be defined before any function uses them)
# ─────────────────────────────────────────────────────────────────────────────
ZONE_FLOOR_COLORS = {
    'North-West': '#c3b1e1', 'North':  '#aec6cf', 'North-East': '#a8d8a8',
    'West':       '#f4a460', 'Center': '#eeeeee', 'East':       '#ffd966',
    'South-West': '#c8a882', 'South':  '#ff9999', 'South-East': '#ffb347',
}
ZONE_SHORT = {
    'North-East': 'NE', 'North-West': 'NW',
    'South-East': 'SE', 'South-West': 'SW',
    'North': 'N', 'South': 'S', 'East': 'E', 'West': 'W', 'Center': 'C',
}
ZONE_COMPASS = {
    'North-East': 'North-East corner (Ishanya) — knowledge & prosperity',
    'North':      'North wall (Kubera) — career & wealth',
    'North-West': 'North-West corner (Vayavya) — movement & travel',
    'East':       'East wall (Surya) — health & new beginnings',
    'Center':     'Center (Brahmasthan) — sacred energy hub',
    'West':       'West wall (Varun) — nourishment & gains',
    'South-East': 'South-East corner (Agni) — fire & energy',
    'South':      'South wall (Yama) — stability',
    'South-West': 'South-West corner (Nairutya) — earth & stability',
}
ZONE_GRID_3D = [
    ('North-West', 0,   0,   1/3, 1/3), ('North',  1/3, 0,   1/3, 1/3),
    ('North-East', 2/3, 0,   1/3, 1/3), ('West',   0,   1/3, 1/3, 1/3),
    ('Center',     1/3, 1/3, 1/3, 1/3), ('East',   2/3, 1/3, 1/3, 1/3),
    ('South-West', 0,   2/3, 1/3, 1/3), ('South',  1/3, 2/3, 1/3, 1/3),
    ('South-East', 2/3, 2/3, 1/3, 1/3),
]
ZONE_FLOOR_LABELS = {
    'North-West': 'NW\n\u2196 North-West', 'North':      'N\n\u2191 North',
    'North-East': 'NE\n\u2197 North-East', 'West':        'W\n\u2190 West',
    'Center':     'C\nCenter',               'East':        'E\nEast \u2192',
    'South-West': 'SW\n\u2199 South-West', 'South':       'S\n\u2193 South',
    'South-East': 'SE\n\u2198 South-East',
}
FURNITURE = {
    'sofa': (2.0, 0.8, 0.9),   'couch': (2.2, 0.8, 0.95),  'bed': (2.0, 0.6, 1.8),
    'tv': (1.2, 0.7, 0.15),    'television': (1.2, 0.7, 0.15),
    'chair': (0.5, 0.9, 0.5),  'stool': (0.4, 0.6, 0.4),   'bench': (1.4, 0.45, 0.5),
    'desk': (1.4, 0.75, 0.7),  'table': (1.4, 0.75, 0.8),
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
    'sofa': [100,150,200,255],  'couch': [100,150,200,255],
    'bed': [150,100,100,255],   'tv': [30,30,30,255],  'television': [30,30,30,255],
    'chair': [139,90,60,255],   'stool': [160,120,80,255],  'bench': [160,130,90,255],
    'desk': [180,140,100,255],  'table': [170,130,90,255],
    'dining table': [160,120,80,255],  'nightstand': [150,110,70,255],
    'cabinet': [140,110,80,255], 'wardrobe': [130,100,70,255],
    'bookcase': [160,130,100,255], 'shelf': [150,120,90,255],
    'dresser': [145,115,85,255],  'cupboard': [140,110,80,255], 'drawer': [150,120,90,255],
    'refrigerator': [220,220,220,255], 'oven': [200,200,200,255],
    'microwave': [180,180,180,255], 'sink': [230,230,230,255],
    'toaster': [190,160,130,255],  'kettle': [200,90,50,255],
    'dishwasher': [210,210,210,255], 'washing machine': [220,220,220,255],
    'dryer': [200,200,200,255],
    'laptop': [60,60,60,255],   'computer': [80,80,80,255],
    'monitor': [50,50,50,255],  'keyboard': [70,70,70,255],
    'cell phone': [40,40,40,255], 'remote': [60,60,60,255],
    'toilet': [255,255,255,255], 'bathtub': [240,240,255,255],
    'shower': [200,230,255,255], 'towel': [200,180,160,255],
    'potted plant': [60,140,60,255], 'vase': [200,150,100,255],
    'clock': [50,50,50,255],    'mirror': [180,210,230,200],
    'book': [180,140,100,255],  'painting': [210,180,140,255],
    'picture frame': [160,130,100,255], 'candle': [255,220,100,255],
    'lamp': [240,220,100,255],  'chandelier': [220,190,80,255], 'light': [255,240,150,255],
    'curtain': [210,180,160,200], 'carpet': [160,110,80,200],
    'rug': [170,120,90,200],    'blanket': [200,180,160,255],
    'pillow': [220,200,180,255], 'cushion': [190,170,150,255],
    'bicycle': [80,80,200,255], 'umbrella': [100,180,100,255],
    'backpack': [100,120,180,255], 'suitcase': [120,100,80,255], 'bag': [110,90,70,255],
    'fan': [150,190,210,255],   'air conditioner': [180,210,230,255],
    'trash can': [100,100,100,255], 'dustbin': [100,100,100,255], 'bin': [100,100,100,255],
    'door': [180,140,100,255],  'window': [180,220,240,200],
    'flower': [220,100,130,255],
}


# ── Furniture emoji icons ──────────────────────────────────────────────────────
FURN_EMOJI = {
    'bed':          '🛏️',
    'wardrobe':     '🚪',
    'desk':         '🖥️',
    'tv':           '📺',
    'dustbin':      '🗑️',
    'chair':        '🪑',
    'sofa':         '🛋️',
    'potted plant': '🪴',
    'dining table': '🍽️',
    'refrigerator': '🧊',
    'microwave':    '📦',
    'sink':         '🚿',
    'oven':         '🍳',
    'kettle':       '☕',
    'toilet':       '🚽',
    'mirror':       '🪞',
    'bathtub':      '🛁',
    'lamp':         '💡',
    'computer':     '💻',
    'cabinet':      '🗄️',
    'door':         '🚪',
    'window':       '🪟',
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper: determine ghost status correctly
# ─────────────────────────────────────────────────────────────────────────────
def _is_ghost(item: dict, is_current: bool) -> bool:
    """
    Ghost = cyan outline meaning "this item is expected but NOT in the video".
    Only True when ALL three conditions hold:
      1. is_current=True   (we are drawing the current layout)
      2. is_essential=True (user selected this furniture type)
      3. not_detected_in_video=True (YOLO did NOT find it in the video)

    If item was detected by YOLO → ghost=False (show solid box) regardless
    of whether it is also marked essential.
    """
    if not is_current:
        return False
    if not item.get('is_essential', False):
        return False
    return item.get('not_detected_in_video', False)


# ─────────────────────────────────────────────────────────────────────────────
# Build current_list: merge DETECTIONS + RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────
def _build_current_list(vastu_result: dict) -> list:
    """
    CURRENT LAYOUT = ONLY furniture physically detected by YOLO in the video.
    is_essential=False means YOLO found it but user did not select it.
    is_essential=True  means user selected it AND YOLO found it (both).

    Items that are ONLY user-selected (not detected by YOLO) are excluded —
    they appear in the optimised layout instead.
    """
    recs = vastu_result.get('recommendations', [])

    print(f"[BUILD_CURRENT] {len(recs)} recs from pipeline:")
    for r in recs:
        print(f"  {r.get('object')} | is_essential={r.get('is_essential')} | "
              f"current_zone={r.get('current_zone')}")

    # Include all items that have a real detected position
    # Exclude items that are essential-only (user selected but NOT detected by YOLO)
    # The pipeline marks not_detected_in_video=True for those, or we can infer
    # from is_essential=True with no YOLO confidence (conf_score == 15 = neutral)
    current_list = []
    for r in recs:
        name = r.get('object', '')
        if not name:
            continue
        is_ess       = bool(r.get('is_essential', False))
        not_detected = bool(r.get('not_detected_in_video', False))
        # Skip items that user selected but YOLO never found in the video
        if is_ess and not_detected:
            continue
        current_list.append({
            'object':        name,
            'position_3d':   list(r.get('current_position_3d') or
                                  r.get('position_3d') or [0.0, 0.0, 0.0]),
            'zone':          r.get('current_zone') or r.get('zone', ''),
            'is_essential':  is_ess,
            'action_needed': bool(r.get('action_needed', False)),
            'not_detected_in_video': False,
        })

    print(f"[BUILD_CURRENT] → {len(current_list)} detected items: "
          + ", ".join(f"{i['object']}(ess={i['is_essential']})" for i in current_list))
    return current_list
def _hex_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))

def _media_url(p):
    if not p or not os.path.exists(p): return ''
    rel = os.path.relpath(p, MEDIA_ROOT)
    # Normalize Windows backslashes to forward slashes for URLs
    return '/media/' + rel.replace(os.sep, '/')


# ─────────────────────────────────────────────────────────────────────────────
# UNet
# ─────────────────────────────────────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
            nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True))
    def forward(self, x): return self.net(x)

class UNetSegmentation(nn.Module):
    def __init__(self, in_channels=4, num_classes=10):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 64); self.enc2 = DoubleConv(64,  128)
        self.enc3 = DoubleConv(128, 256);         self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2); self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512,  256, 2, stride=2); self.dec3 = DoubleConv(512,  256)
        self.up2 = nn.ConvTranspose2d(256,  128, 2, stride=2); self.dec2 = DoubleConv(256,  128)
        self.up1 = nn.ConvTranspose2d(128,   64, 2, stride=2); self.dec1 = DoubleConv(128,   64)
        self.final = nn.Conv2d(64, num_classes, 1)
    def forward(self, img, depth):
        if depth.dim() == 3: depth = depth.unsqueeze(1)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        x  = torch.cat([img, depth], dim=1)
        e1 = self.enc1(x);               e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2));   e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)

def _load_unet():
    global _UNET
    if _UNET: return _UNET
    for root, _, files in os.walk(ML_MODELS_DIR):
        for f in files:
            if f.endswith('.pth'):
                pth = os.path.join(root, f)
                m = UNetSegmentation(4, 10)
                s = torch.load(pth, map_location=DEVICE)
                if any(k.startswith('module.') for k in s):
                    s = {k.replace('module.', ''): v for k, v in s.items()}
                m.load_state_dict(s, strict=False)
                m.to(DEVICE).eval()
                _UNET = m
                logger.info(f"UNet loaded: {pth}")
                return _UNET
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 3D draw helpers
# ─────────────────────────────────────────────────────────────────────────────
def _draw_zone_floor(ax, rw, rd):
    hw = rw / 2
    for zone, cx, cz, wf, df in ZONE_GRID_3D:
        x0 = -hw + cx*rw;  x1 = x0 + wf*rw
        z0 = cz*rd;         z1 = z0 + df*rd
        mx = (x0+x1)/2;     mz = (z0+z1)/2
        col     = _hex_rgb(ZONE_FLOOR_COLORS.get(zone, '#eeeeee'))
        col_hex = ZONE_FLOOR_COLORS.get(zone, '#eeeeee')
        ax.add_collection3d(Poly3DCollection(
            [[[x0,0,z0],[x1,0,z0],[x1,0,z1],[x0,0,z1]]],
            alpha=0.70, facecolor=col, edgecolor='#ffffff', linewidth=1.2))
        lbl = ZONE_FLOOR_LABELS.get(zone, ZONE_SHORT.get(zone, zone))
        ax.text(mx, 0.01, mz, lbl, fontsize=7, ha='center', va='bottom',
                color='#111111', fontweight='bold', linespacing=1.4,
                bbox=dict(boxstyle='round,pad=0.22', facecolor='white',
                          alpha=0.82, edgecolor=col_hex, linewidth=1.5))


def _draw_walls(ax, rw, rd, ch=2.5):
    hw = rw / 2
    for txt, x, y, z, ha, col in [
        ('N \u2191 NORTH',  0,      0.3, -0.4,    'center', '#4488ff'),
        ('SOUTH \u2193 S',  0,      0.3, rd+0.35, 'center', '#ff4444'),
        ('W \u2190 WEST', -hw-0.2,  0.3, rd/2,    'right',  '#ff9900'),
        ('EAST \u2192 E',   hw+0.2, 0.3, rd/2,    'left',   '#00aa44'),
    ]:
        ax.text(x, y, z, txt, fontsize=11, color=col, ha=ha, va='center',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#12122a',
                          alpha=0.90, edgecolor=col, linewidth=2.0))


def _draw_box(ax, number, name, pos3d, zone, is_ess, bad, ghost=False):
    """
    Visual categories
    ─────────────────
    ghost   : cyan outline, faint fill  → essential item NOT found in video
    is_ess  : solid teal                → essential item found in video (or optimised)
    bad     : dark red                  → Vastu violation
    default : palette colour            → detected, compliant
    """
    X, _, Z = pos3d
    W2, H2, D2 = FURNITURE.get(name, (0.8, 0.8, 0.8))
    x0, x1 = X - W2/2, X + W2/2
    y0, y1 = 0, H2
    z0, z1 = Z - D2/2, Z + D2/2

    faces = [
        [[x0,y0,z0],[x1,y0,z0],[x1,y0,z1],[x0,y0,z1]],
        [[x0,y1,z0],[x1,y1,z0],[x1,y1,z1],[x0,y1,z1]],
        [[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0]],
        [[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]],
        [[x0,y0,z0],[x0,y0,z1],[x0,y1,z1],[x0,y1,z0]],
        [[x1,y0,z0],[x1,y0,z1],[x1,y1,z1],[x1,y1,z0]],
    ]

    if ghost:
        face_col, alpha, edge, lw = (0.0, 0.85, 0.85), 0.15, '#00FFFF', 2.0
    elif is_ess:
        face_col, alpha, edge, lw = (0.0, 0.72, 0.65), 0.92, '#00FFD0', 2.8
    elif bad:
        face_col, alpha, edge, lw = (0.45, 0.05, 0.05), 0.88, '#FF4444', 2.2
    else:
        c = FURN_COLORS.get(name, [160, 160, 160, 255])
        face_col = (c[0]/255, c[1]/255, c[2]/255)
        alpha, edge, lw = 0.88, '#7788aa', 1.0

    ax.add_collection3d(Poly3DCollection(
        faces, alpha=alpha, facecolor=face_col, edgecolor=edge, linewidth=lw))

    if ghost:     num_bg, num_fc = '#005566', '#00FFFF'
    elif is_ess:  num_bg, num_fc = '#00FFD0', '#000000'
    elif bad:     num_bg, num_fc = '#FF4444', '#ffffff'
    else:         num_bg, num_fc = '#3b2f8f', '#ffffff'

    ax.text(X, y1+0.18, Z, f' {number} ',
            fontsize=8, ha='center', va='bottom', color=num_fc, fontweight='bold',
            bbox=dict(boxstyle='circle,pad=0.3', facecolor=num_bg,
                      alpha=1.0, edgecolor='white', linewidth=1.2))

    zone_s = ZONE_SHORT.get(zone, zone)
    if ghost:    sfx, lbg, lfc = ' ➕ ADD', '#003344', '#00FFFF'
    elif is_ess: sfx, lbg, lfc = ' ★',     '#003d35', '#00FFD0'
    elif bad:    sfx, lbg, lfc = ' ⚠',     '#5a0000', '#FF9999'
    else:        sfx, lbg, lfc = '',        '#1a1a2e', '#e0e0e0'

    ax.text(X, y1+0.06, Z, f"{name}\n[{zone_s}]{sfx}",
            fontsize=6, ha='center', va='top', color=lfc, linespacing=1.35,
            bbox=dict(boxstyle='round,pad=0.22', facecolor=lbg,
                      alpha=0.90, edgecolor=edge, linewidth=0.6))


# Zone centre points in normalised (cx, cy) coords — used for room placement
_ZONE_NORM_CENTERS = {
    'North-West': (0.165, 0.165), 'North':  (0.500, 0.165), 'North-East': (0.835, 0.165),
    'West':       (0.165, 0.500), 'Center': (0.500, 0.500), 'East':       (0.835, 0.500),
    'South-West': (0.165, 0.835), 'South':  (0.500, 0.835), 'South-East': (0.835, 0.835),
}

def _remap_position(pos3d, rw, rd, zone='', index=0, total=1):
    """
    Convert pipeline position to room 3D coords guaranteed inside room.

    Strategy (in priority order):
    1. Normalised coords (0-1 range): scale directly to room metres.
    2. Zone-based: use the item's Vastu zone to find its cell centre,
       then offset slightly by index so items don't pile up.
    3. Camera-space fallback: shift+clamp into room bounds.
    """
    X, Y, Z = pos3d
    hw = rw / 2

    # Strategy 1: normalised coords
    if 0.0 <= X <= 1.0 and 0.0 <= Z <= 1.0 and abs(Y) < 0.5:
        rx = float(X * rw - hw)
        rz = float(Z * rd)
        return rx, 0.0, rz

    # Strategy 2: zone-based placement (most reliable — every item in its zone)
    if zone and zone in _ZONE_NORM_CENTERS:
        cx_n, cy_n = _ZONE_NORM_CENTERS[zone]
        # Spread items within the zone cell using a small grid offset
        cols = max(1, int(total ** 0.5))
        row  = index // cols
        col  = index %  cols
        step = 0.06
        cx_n = cx_n + (col - cols/2) * step
        cy_n = cy_n + (row - cols/2) * step
        rx   = float(np.clip(cx_n * rw - hw, -hw + 0.3, hw - 0.3))
        rz   = float(np.clip(cy_n * rd,        0.3,       rd - 0.3))
        return rx, 0.0, rz

    # Strategy 3: camera-space fallback — shift by hw then clamp
    margin = 0.4
    X_room = float(np.clip(X + hw, margin, rw - margin)) - hw
    Z_depth_norm = min(max(Z, 2.0), 8.0)                    # depth 2-8m typical
    Z_room = float((Z_depth_norm - 2.0) / 6.0 * rd)         # map to 0..rd
    Z_room = float(np.clip(Z_room, margin, rd - margin))
    return X_room, 0.0, Z_room


def _render_layout(ax, title, furn_list, room_dims, ess_set, is_current=False):
    rw, rd, ch = room_dims
    _draw_zone_floor(ax, rw, rd)
    _draw_walls(ax, rw, rd, ch)
    n = len(furn_list)
    print(f"[RENDER_LAYOUT] '{title}' — {n} items, room={rw:.1f}x{rd:.1f}x{ch:.1f}m")
    for i, item in enumerate(furn_list):
        is_ess  = item.get('is_essential', False) or item['object'] in ess_set
        bad     = item.get('action_needed', False)
        ghost   = _is_ghost(item, is_current)
        raw_pos = tuple(item['position_3d'])
        zone    = item.get('zone', '')
        pos     = _remap_position(raw_pos, rw, rd, zone=zone, index=i, total=n)
        print(f"  [{i+1}] {item['object']:20} raw={[round(v,2) for v in raw_pos]} "
              f"zone={zone} → pos=({pos[0]:.2f}, {pos[2]:.2f})")
        _draw_box(ax, i + 1, item['object'], pos,
                  zone, is_ess, bad, ghost=ghost)
    hw = rw / 2
    ax.set_xlim(-hw-0.5, hw+0.5)
    ax.set_ylim(-0.2, ch+1.0)
    ax.set_zlim(-0.5, rd+0.5)
    ax.set_xlabel('West ← → East',  fontsize=8, color='#cccccc', labelpad=5)
    ax.set_ylabel('Height (m)',       fontsize=8, color='#cccccc', labelpad=5)
    ax.set_zlabel('North ← → South', fontsize=8, color='#cccccc', labelpad=5)
    ax.tick_params(colors='#888888', labelsize=6)
    ax.set_facecolor('#12122a')
    for p in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        p.set_facecolor('#0d0d1a'); p.set_edgecolor('#3b2f8f'); p.fill = True
    ax.set_title(title, fontsize=13, fontweight='bold', color='white', pad=14)
    ax.view_init(elev=25, azim=35)


def _make_legend():
    patches = [
        mpatches.Patch(facecolor=col, edgecolor='#888', linewidth=.5,
                        label=f"{ZONE_SHORT[zone]} = {zone}")
        for zone, col in ZONE_FLOOR_COLORS.items()
    ]
    patches += [
        mpatches.Patch(facecolor=(0.0,0.72,0.65), edgecolor='#00FFD0',
                        linewidth=2.8, label='★ Essential (user-selected)'),
        mpatches.Patch(facecolor=(0.45,0.05,0.05), edgecolor='#FF4444',
                        linewidth=2.0, label='⚠ Vastu violation'),
        mpatches.Patch(facecolor='#7788aa', edgecolor='#7788aa',
                        linewidth=0.7, label='✓ Detected in video'),
    ]
    return patches


def _draw_numbered_item_legend(ax_leg, furn_list, is_current=False):
    ax_leg.text(0.04, 0.30, 'Item Index:',
                transform=ax_leg.transAxes,
                fontsize=8, color='white', fontweight='bold', va='top')
    row_h = 0.052
    for i, item in enumerate(furn_list):
        y     = 0.27 - i * row_h
        if y < 0.01: break
        name   = item['object']
        is_ess = item.get('is_essential', False)
        bad    = item.get('action_needed', False)
        zone   = item.get('zone', '')
        ghost  = _is_ghost(item, is_current)

        if ghost:    swatch_col, edge_col = (0.0,0.85,0.85), '#00FFFF'
        elif is_ess: swatch_col, edge_col = (0.0,0.72,0.65), '#00FFD0'
        elif bad:    swatch_col, edge_col = (0.45,0.05,0.05), '#FF4444'
        else:
            c = FURN_COLORS.get(name, [160,160,160,255])
            swatch_col, edge_col = (c[0]/255, c[1]/255, c[2]/255), '#7788aa'

        ax_leg.add_patch(plt.Rectangle(
            (0.04, y-0.034), 0.065, 0.034, transform=ax_leg.transAxes,
            facecolor=swatch_col, alpha=0.25 if ghost else 0.9,
            edgecolor=edge_col, linewidth=1.2, zorder=3))

        if ghost:    num_bg, num_fc = '#005566', '#00FFFF'
        elif is_ess: num_bg, num_fc = '#00FFD0', '#000000'
        elif bad:    num_bg, num_fc = '#FF4444', '#ffffff'
        else:        num_bg, num_fc = '#3b2f8f', '#ffffff'

        ax_leg.text(0.072, y-0.015, str(i+1), transform=ax_leg.transAxes,
                    fontsize=6, color=num_fc, fontweight='bold',
                    ha='center', va='center', zorder=4,
                    bbox=dict(boxstyle='circle,pad=0.12',
                              facecolor=num_bg, edgecolor='white', linewidth=0.7))

        if ghost:    sfx, fc = ' ➕ ADD', '#00FFFF'
        elif is_ess: sfx, fc = ' ★',     '#00FFD0'
        elif bad:    sfx, fc = ' ⚠',     '#FF9999'
        else:        sfx, fc = '',        '#dddddd'

        ax_leg.text(0.12, y-0.015, f"{name}  [{ZONE_SHORT.get(zone, zone)}]{sfx}",
                    transform=ax_leg.transAxes,
                    fontsize=6.5, color=fc, va='center', ha='left')


# ─────────────────────────────────────────────────────────────────────────────
# Direction index image
# ─────────────────────────────────────────────────────────────────────────────
def _generate_direction_index(furn_list: list, out_dir: str,
                               tag: str = 'current') -> str:
    fig, ax = plt.subplots(figsize=(14, max(4, 0.45*len(furn_list)+2)),
                            facecolor='#12122a')
    ax.set_facecolor('#12122a')
    ax.axis('off')
    ax.set_title(
        'Direction Index — Current Layout' if tag == 'current'
        else 'Direction Index — Vastu-Optimised Layout',
        fontsize=14, fontweight='bold', color='white', pad=12)

    cols  = ['#', 'Furniture', 'Zone', 'Compass Direction', 'Type', 'Status']
    col_w = [0.04, 0.16, 0.10, 0.38, 0.14, 0.13]
    hdr_y = 0.97

    ax.add_patch(plt.Rectangle((0, hdr_y-0.05), 1, 0.06,
                                transform=ax.transAxes,
                                facecolor='#3b2f8f', zorder=2))
    x = 0.01
    for col, w in zip(cols, col_w):
        ax.text(x, hdr_y, col, transform=ax.transAxes,
                fontsize=9, fontweight='bold', color='white',
                va='top', ha='left', zorder=3)
        x += w

    row_h = 0.075
    for i, item in enumerate(furn_list):
        y      = hdr_y - 0.07 - i*row_h
        name   = item['object']
        zone   = item['zone']
        is_ess = item.get('is_essential', False)
        bad    = item.get('action_needed', False)
        ghost  = _is_ghost(item, tag == 'current')
        compass = ZONE_COMPASS.get(zone, zone)
        ftype  = ('➕ Add to room' if ghost
                  else '★ Essential' if is_ess else '✓ Detected')
        s_txt  = '⚠ Move needed' if bad else '✓ Compliant'

        row_col = '#1a1a2e' if i % 2 == 0 else '#1e1e3a'
        ax.add_patch(plt.Rectangle((0, y-row_h+0.01), 1, row_h-0.005,
                                    transform=ax.transAxes,
                                    facecolor=row_col, zorder=1))
        bar_col = '#FFD700' if is_ess else ('#FF4444' if bad else '#4CAF50')
        ax.add_patch(plt.Rectangle((0, y-row_h+0.01), 0.004, row_h-0.005,
                                    transform=ax.transAxes,
                                    facecolor=bar_col, zorder=3))

        zone_col = ZONE_FLOOR_COLORS.get(zone, '#eeeeee')
        x = 0.01
        vals      = [str(i+1), name, zone, compass, ftype, s_txt]
        text_cols = [
            '#aaaaaa',
            '#00FFFF' if ghost else ('#00FFD0' if is_ess else '#e2e8f0'),
            '#ffffff', '#94a3b8',
            '#00FFFF' if ghost else ('#00FFD0' if is_ess else '#60a5fa'),
            '#f87171' if bad else '#4ade80',
        ]
        for val, w, tc in zip(vals, col_w, text_cols):
            if val == zone:
                ax.add_patch(plt.Rectangle((x, y-0.028), 0.012, 0.028,
                                            transform=ax.transAxes,
                                            facecolor=zone_col, zorder=4,
                                            linewidth=0.5, edgecolor='#888'))
                ax.text(x+0.015, y-0.005, val, transform=ax.transAxes,
                        fontsize=8, color=tc, va='top', ha='left', zorder=5)
            else:
                ax.text(x, y-0.005, val, transform=ax.transAxes,
                        fontsize=8, color=tc, va='top', ha='left', zorder=5)
            x += w

    plt.tight_layout()
    fpath = os.path.join(out_dir, f'direction_index_{tag}.png')
    plt.savefig(fpath, dpi=150, bbox_inches='tight', facecolor='#12122a')
    plt.close(fig)
    logger.info(f"Direction index saved → {fpath}")
    return fpath


# ─────────────────────────────────────────────────────────────────────────────
# Interactive Three.js HTML
# ─────────────────────────────────────────────────────────────────────────────
def _generate_interactive_html(current_list: list, opt_list: list,
                                room_dims: tuple, out_dir: str,
                                cmp_stats: dict = None) -> str:
    """
    Generates ONE self-contained HTML page with 3 tabs:
      Tab 1 - Current Layout  (Three.js 3D, rotatable)
      Tab 2 - Vastu-Optimised (Three.js 3D, rotatable)
      Tab 3 - Comparison stats
    Uses StringIO buffer to write HTML directly — avoids ALL Python
    f-string / template-literal / HTML-tag escaping issues.
    """
    import io
    rw, rd, ch = room_dims
    n_cur = len(current_list)
    n_opt = len(opt_list)

    def _ser(lst, n, is_current):
        out = []
        for i, item in enumerate(lst):
            name   = item['object']
            is_ess = item.get('is_essential', False)
            bad    = item.get('action_needed', False)
            fw, fh, fd = FURNITURE.get(name, (0.8, 0.8, 0.8))
            c      = FURN_COLORS.get(name, [120, 140, 180, 255])
            zone_i = item.get('zone', '')
            raw    = item.get('position_3d', [0, 0, 0])
            rx, _, rz = _remap_position(tuple(raw), rw, rd,
                                         zone=zone_i, index=i, total=n)
            px = float(rx + rw / 2)
            pz = float(rz)
            if bad:
                hex_col, opacity, cat = '#7a0d0d', 0.88, 'violation'
            elif is_ess:
                hex_col, opacity, cat = '#00b8a5', 0.92, 'essential'
            else:
                hex_col = '#{:02x}{:02x}{:02x}'.format(c[0], c[1], c[2])
                opacity, cat = 0.88, 'detected'
            compass = ZONE_COMPASS.get(zone_i, zone_i)
            emoji   = FURN_EMOJI.get(name, '\U0001f4e6')
            if is_current and bad:   s_txt = '\u26a0 Move needed'
            elif is_ess:             s_txt = '\u2605 Essential'
            elif not is_current:     s_txt = '\u2605 Vastu-correct'
            else:                    s_txt = '\u2713 Detected'
            out.append({
                'id': i+1, 'name': name, 'emoji': emoji,
                'px': px, 'py': 0, 'pz': pz,
                'fw': fw, 'fh': fh, 'fd': fd,
                'color': hex_col, 'opacity': opacity, 'cat': cat,
                'zone': zone_i, 'compass': compass, 'status': s_txt,
            })
        return out

    cur_json   = json.dumps(_ser(current_list, n_cur, True))
    opt_json   = json.dumps(_ser(opt_list,     n_opt, False))
    zcols_json = json.dumps(ZONE_FLOOR_COLORS)
    zcomp_json = json.dumps(ZONE_COMPASS)
    fpath      = os.path.join(out_dir, 'interactive_3d.html')

    w = io.StringIO()  # write to buffer — never use f-string for HTML content

    # ── DOCTYPE + head ───────────────────────────────────────────────────────
    w.write('<!DOCTYPE html>\n<html lang="en">\n<head>\n')
    w.write('<meta charset="UTF-8"/>\n')
    w.write('<meta name="viewport" content="width=device-width,initial-scale=1"/>\n')
    w.write('<title>Vastu 3D Room Viewer</title>\n')
    w.write('<style>\n')
    # Write CSS as a plain triple-quoted string — no interpolation needed
    CSS = (
        '*{margin:0;padding:0;box-sizing:border-box;}\n'
        'body{background:#0d0d1a;font-family:\'Segoe UI\',sans-serif;'
        'height:100vh;display:flex;flex-direction:column;overflow:hidden;}\n'
        '#hdr{background:linear-gradient(135deg,#1a1a3a,#12122a);'
        'border-bottom:2px solid #3b2f8f;padding:10px 20px;'
        'display:flex;align-items:center;gap:14px;flex-shrink:0;}\n'
        '#hdr-title{color:#e2e8f0;font-weight:800;font-size:16px;}\n'
        '#hdr-sub{color:#555;font-size:11px;margin-top:2px;}\n'
        '#close-btn{margin-left:auto;background:#1e1e3a;color:#7c6fe0;'
        'border:1px solid #3b2f8f;border-radius:8px;'
        'padding:8px 18px;cursor:pointer;font-weight:700;font-size:13px;}\n'
        '#close-btn:hover{background:#2a2a4a;}\n'
        '#tabbar{background:#0a0a1e;border-bottom:2px solid #3b2f8f;'
        'display:flex;padding:0 16px;gap:4px;flex-shrink:0;}\n'
        '.tab{padding:10px 20px;background:transparent;color:#555;'
        'border:none;border-bottom:3px solid transparent;'
        'border-radius:8px 8px 0 0;font-weight:700;font-size:13px;'
        'cursor:pointer;display:flex;align-items:center;gap:7px;transition:all .15s;}\n'
        '.tab.active{background:#1a1a3a;}\n'
        '.tab .badge{border-radius:10px;padding:1px 7px;'
        'font-size:10px;font-weight:700;background:#ffffff11;color:#444;}\n'
        '#content{flex:1;overflow:hidden;position:relative;}\n'
        '.panel{display:none;height:100%;flex-direction:column;}\n'
        '.panel.active{display:flex;}\n'
        '.sub-hdr{padding:6px 18px;display:flex;align-items:center;'
        'gap:12px;flex-shrink:0;font-size:11px;color:#555;}\n'
        '.viewer{flex:1;border:none;display:block;background:#0d0d1a;}\n'
        '.hint{background:#080818;padding:4px 18px;font-size:10px;'
        'color:#333;text-align:center;flex-shrink:0;}\n'
        '#cmp-panel{height:100%;overflow-y:auto;'
        'background:linear-gradient(135deg,#12122a,#0d1a10);}\n'
        '.cmp-inner{padding:36px;max-width:860px;margin:0 auto;}\n'
        '.cmp-title{color:#a0e080;font-weight:800;font-size:20px;margin-bottom:28px;}\n'
        '.ratio-row{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin-bottom:28px;}\n'
        '.ratio-card{flex:1;min-width:130px;border-radius:14px;padding:22px;text-align:center;}\n'
        '.ratio-label{font-size:10px;color:#555;margin-bottom:8px;'
        'text-transform:uppercase;letter-spacing:1.5px;}\n'
        '.ratio-num{font-size:56px;font-weight:900;line-height:1;}\n'
        '.ratio-sub{font-size:11px;color:#555;margin-top:8px;}\n'
        '.arrow{font-size:32px;}\n'
        '.stat-grid{display:grid;'
        'grid-template-columns:repeat(auto-fit,minmax(140px,1fr));'
        'gap:14px;margin-bottom:24px;}\n'
        '.stat-card{background:#1e1e3a;border-radius:12px;padding:16px;text-align:center;}\n'
        '.stat-icon{font-size:26px;margin-bottom:8px;}\n'
        '.stat-num{font-size:32px;font-weight:800;}\n'
        '.stat-lbl{font-size:11px;color:#555;margin-top:5px;}\n'
        '.prog-box{background:#1e1e3a;border-radius:12px;padding:20px;}\n'
        '.prog-hdr{display:flex;justify-content:space-between;'
        'font-size:12px;color:#666;margin-bottom:8px;}\n'
        '.prog-bar{position:relative;height:16px;background:#0d0d1a;'
        'border-radius:8px;overflow:hidden;margin-bottom:8px;}\n'
        '.prog-fill{position:absolute;left:0;top:0;height:100%;border-radius:8px;}\n'
        '.prog-note{text-align:center;font-size:12px;font-weight:700;}\n'
    )
    w.write(CSS)
    w.write('</style>\n</head>\n<body>\n')

    # ── Header ───────────────────────────────────────────────────────────────
    w.write('<div id="hdr">\n')
    w.write('  <span style="font-size:22px">&#127968;</span>\n')
    w.write('  <div>\n')
    w.write('    <div id="hdr-title">Vastu 3D Room Viewer</div>\n')
    w.write('    <div id="hdr-sub">Drag to rotate &middot; Scroll to zoom &middot; Click sidebar items to highlight</div>\n')
    w.write('  </div>\n')
    w.write('  <button id="close-btn" onclick="window.close()">&#10005; Close</button>\n')
    w.write('</div>\n')

    # ── Tab bar ──────────────────────────────────────────────────────────────
    w.write('<div id="tabbar">\n')
    w.write('  <button class="tab active" style="color:#7c6fe0;border-bottom-color:#7c6fe0" onclick="switchTab(\'cur\',this)">\n')
    w.write('    &#127968; Current Layout <span class="badge" id="cur-badge">0 detected</span>\n')
    w.write('  </button>\n')
    w.write('  <button class="tab" style="color:#555" onclick="switchTab(\'opt\',this)">\n')
    w.write('    &#10024; Vastu-Optimised <span class="badge" id="opt-badge">0 items</span>\n')
    w.write('  </button>\n')
    w.write('  <button class="tab" style="color:#555" onclick="switchTab(\'cmp\',this)">\n')
    w.write('    &#128202; Comparison <span class="badge" id="cmp-badge">0%</span>\n')
    w.write('  </button>\n')
    w.write('</div>\n')

    # ── Content ──────────────────────────────────────────────────────────────
    w.write('<div id="content">\n')
    w.write('  <div class="panel active" id="panel-cur">\n')
    w.write('    <div class="sub-hdr" style="background:#0a0a1e;border-bottom:1px solid #3b2f8f22">Furniture detected by YOLO at their current positions.</div>\n')
    w.write('    <iframe id="iframe-cur" class="viewer" sandbox="allow-scripts" title="Current 3D"></iframe>\n')
    w.write('    <div class="hint">&#128433; Drag: rotate &nbsp;&middot;&nbsp; Scroll: zoom &nbsp;&middot;&nbsp; Right-drag: pan &nbsp;&middot;&nbsp; Click sidebar to highlight</div>\n')
    w.write('  </div>\n')
    w.write('  <div class="panel" id="panel-opt">\n')
    w.write('    <div class="sub-hdr" style="background:#0a1a15;border-bottom:1px solid #00b8a522">All furniture repositioned to Vastu-correct zones by the genetic algorithm.</div>\n')
    w.write('    <iframe id="iframe-opt" class="viewer" sandbox="allow-scripts" title="Optimised 3D"></iframe>\n')
    w.write('    <div class="hint">&#128433; Drag: rotate &nbsp;&middot;&nbsp; Scroll: zoom &nbsp;&middot;&nbsp; Right-drag: pan &nbsp;&middot;&nbsp; Click sidebar to highlight</div>\n')
    w.write('  </div>\n')
    w.write('  <div class="panel" id="panel-cmp">\n')
    w.write('    <div id="cmp-panel"><div class="cmp-inner" id="cmp-content"></div></div>\n')
    w.write('  </div>\n')
    w.write('</div>\n')

    # ── Main script: data + tab switching + 3D builder + comparison ──────────
    w.write('<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>\n')
    w.write('<script>\n')

    # Inject Python data as JS vars
    w.write('var ROOM_W=' + str(rw) + ',ROOM_D=' + str(rd) + ',ROOM_H=' + str(ch) + ';\n')
    w.write('var ZCOLS=' + zcols_json + ';\n')
    w.write('var ZCOMP=' + zcomp_json + ';\n')
    w.write('var CUR_DATA=' + cur_json + ';\n')
    w.write('var OPT_DATA=' + opt_json + ';\n')
    w.write('var ZGRID=[["North-West",0,0,1/3,1/3],["North",1/3,0,1/3,1/3],["North-East",2/3,0,1/3,1/3],["West",0,1/3,1/3,1/3],["Center",1/3,1/3,1/3,1/3],["East",2/3,1/3,1/3,1/3],["South-West",0,2/3,1/3,1/3],["South",1/3,2/3,1/3,1/3],["South-East",2/3,2/3,1/3,1/3]];\n')
    w.write('var DIRMAP={"North":"\u2191N","South":"\u2193S","East":"\u2192E","West":"\u2190W","North-East":"\u2197NE","North-West":"\u2196NW","South-East":"\u2198SE","South-West":"\u2199SW","Center":"\u2295C"};\n')
    w.write('var DIRCOL={"North":"#4488ff","North-East":"#4488ff","North-West":"#4488ff","South":"#ff4444","South-East":"#ff4444","South-West":"#ff4444","East":"#00cc55","West":"#ffaa00","Center":"#aaa"};\n')

    # Plain JS — written as a Python string literal (no HTML, no interpolation issues)
    JS = r"""
function switchTab(key,btn){
  document.querySelectorAll('.tab').forEach(function(t){
    t.classList.remove('active');t.style.color='#555';t.style.borderBottomColor='transparent';
  });
  document.querySelectorAll('.panel').forEach(function(p){p.classList.remove('active');});
  btn.classList.add('active');
  var colors={cur:'#7c6fe0',opt:'#00b8a5',cmp:'#4caf50'};
  btn.style.color=colors[key]||'#7c6fe0';
  btn.style.borderBottomColor=colors[key]||'#7c6fe0';
  document.getElementById('panel-'+key).classList.add('active');
}

function viewer3DScript(){
  return [
    'var DATA=__DATA__;var ZCOLS_=__ZCOLS__;var ZGRID_=__ZGRID__;var DIRMAP_=__DIRMAP__;var DIRCOL_=__DIRCOL__;var RW=__RW__;var RD=__RD__;',
    'var cvEl=document.getElementById("cv");',
    'var W=function(){return cvEl.clientWidth;},H=function(){return window.innerHeight;};',
    'var renderer=new THREE.WebGLRenderer({antialias:true});',
    'renderer.setPixelRatio(window.devicePixelRatio);renderer.setSize(W(),H());renderer.shadowMap.enabled=true;',
    'cvEl.appendChild(renderer.domElement);',
    'var scene=new THREE.Scene();scene.background=new THREE.Color(0x0d0d1a);scene.fog=new THREE.FogExp2(0x0d0d1a,.038);',
    'var camera=new THREE.PerspectiveCamera(45,W()/H(),.1,120);',
    'scene.add(new THREE.AmbientLight(0xffffff,.52));',
    'var dl=new THREE.DirectionalLight(0xffffff,.88);dl.position.set(6,10,6);dl.castShadow=true;scene.add(dl);',
    'scene.add(new THREE.HemisphereLight(0x334466,0x0d0d1a,.42));',
    'var drag=false,rDrag=false,pm={x:0,y:0};',
    'var theta=0.55,phi=1.05,rad=Math.sqrt(RW*RW+RD*RD)*1.9;',
    'var panX=RW/2,panY=0,panZ=RD/2;',
    'function camUp(){camera.position.set(panX+rad*Math.sin(phi)*Math.sin(theta),panY+rad*Math.cos(phi),panZ+rad*Math.sin(phi)*Math.cos(theta));camera.lookAt(panX,panY,panZ);}camUp();',
    'cvEl.addEventListener("mousedown",function(e){drag=true;rDrag=e.button===2;pm={x:e.clientX,y:e.clientY};});',
    'cvEl.addEventListener("contextmenu",function(e){e.preventDefault();});',
    'window.addEventListener("mouseup",function(){drag=false;});',
    'window.addEventListener("mousemove",function(e){if(!drag)return;var dx=(e.clientX-pm.x)*.005,dy=(e.clientY-pm.y)*.005;if(rDrag){var r=new THREE.Vector3();r.crossVectors(camera.getWorldDirection(new THREE.Vector3()),new THREE.Vector3(0,1,0)).normalize();panX-=r.x*dx*3;panZ-=r.z*dx*3;panY+=dy*3;}else{theta-=dx;phi=Math.max(.08,Math.min(Math.PI/2-.03,phi+dy));}pm={x:e.clientX,y:e.clientY};camUp();});',
    'cvEl.addEventListener("wheel",function(e){rad=Math.max(1.5,Math.min(40,rad+e.deltaY*.025));camUp();});',
    'function mkSp(txt,o){o=o||{};var fs=o.fs||48,col=o.col||"#fff",bg=o.bg||"rgba(0,0,0,0)";var cv=document.createElement("canvas");cv.width=256;cv.height=128;var x=cv.getContext("2d");x.fillStyle=bg;x.fillRect(0,0,256,128);x.font="bold "+fs+"px Segoe UI Emoji,Segoe UI,sans-serif";x.fillStyle=col;x.textAlign="center";x.textBaseline="middle";x.fillText(txt,128,64);var sp=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(cv),transparent:true,depthTest:false}));sp.scale.set(o.sw||.9,o.sh||.45,1);return sp;}',
    'function buildFloor(){ZGRID_.forEach(function(g){var zone=g[0],cx=g[1],cz=g[2],wf=g[3],df=g[4];var zw=wf*RW,zd=df*RD,x0=cx*RW,z0=cz*RD;var col=ZCOLS_[zone]||"#ccc";var m=new THREE.Mesh(new THREE.PlaneGeometry(zw-.04,zd-.04),new THREE.MeshLambertMaterial({color:new THREE.Color(col),transparent:true,opacity:.70,side:THREE.DoubleSide}));m.rotation.x=-Math.PI/2;m.position.set(x0+zw/2,.001,z0+zd/2);scene.add(m);var el=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(zw,zd)),new THREE.LineBasicMaterial({color:0xffffff,transparent:true,opacity:.16}));el.rotation.x=-Math.PI/2;el.position.set(x0+zw/2,.002,z0+zd/2);scene.add(el);var sh=zone.replace("North","N").replace("South","S").replace("East","E").replace("West","W").replace("-","");var sp=mkSp(sh,{fs:26,col:"#111",sw:.58,sh:.28});sp.position.set(x0+zw/2,.04,z0+zd/2);scene.add(sp);});}',
    'function buildCompass(){[["N",RW/2,1.4,-.4,"#4488ff"],["S",RW/2,1.4,RD+.4,"#ff4444"],["W",-.4,1.4,RD/2,"#ffaa00"],["E",RW+.4,1.4,RD/2,"#00cc55"]].forEach(function(a){var t=a[0],x=a[1],y=a[2],z=a[3],col=a[4];var cv=document.createElement("canvas");cv.width=120;cv.height=120;var ctx=cv.getContext("2d");ctx.fillStyle="rgba(8,6,28,.94)";ctx.beginPath();ctx.arc(60,60,57,0,Math.PI*2);ctx.fill();ctx.strokeStyle=col;ctx.lineWidth=4;ctx.beginPath();ctx.arc(60,60,55,0,Math.PI*2);ctx.stroke();ctx.font="bold 58px Segoe UI,sans-serif";ctx.fillStyle=col;ctx.textAlign="center";ctx.textBaseline="middle";ctx.fillText(t,60,63);var sp=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(cv),transparent:true,depthTest:false}));sp.scale.set(.8,.8,1);sp.position.set(x,y,z);scene.add(sp);});}',
    'var fGrp=new THREE.Group();scene.add(fGrp);',
    'var ray=new THREE.Raycaster(),mV=new THREE.Vector2(-9,-9);',
    'var mmap={},bmap={};var focId=null;',
    'DATA.forEach(function(item){var hw=RW/2;var px=item.px-hw,py=item.fh/2,pz=item.pz;var geo=new THREE.BoxGeometry(item.fw,item.fh,item.fd);var mat=new THREE.MeshLambertMaterial({color:new THREE.Color(item.color),transparent:true,opacity:item.opacity});var box=new THREE.Mesh(geo,mat);box.position.set(px,py,pz);box.castShadow=true;fGrp.add(box);mmap[box.uuid]=item;var ec=item.cat==="essential"?0x00ffd0:item.cat==="violation"?0xff4444:0x7788aa;box.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo),new THREE.LineBasicMaterial({color:ec,linewidth:1.5})));var oGeo=new THREE.BoxGeometry(item.fw*1.07,item.fh*1.07,item.fd*1.07);var oMat=new THREE.MeshBasicMaterial({color:0x00ffd0,side:THREE.BackSide,transparent:true,opacity:0});var oMesh=new THREE.Mesh(oGeo,oMat);box.add(oMesh);var es=mkSp(item.emoji,{fs:44,sw:.54,sh:.54});es.position.set(0,item.fh/2+.02,0);box.add(es);var nBg=item.cat==="essential"?"rgba(0,255,208,.95)":item.cat==="violation"?"rgba(180,20,20,.9)":"rgba(59,47,143,.9)";var nFc=item.cat==="essential"?"#000":"#fff";var ns=mkSp(String(item.id),{fs:32,col:nFc,bg:nBg,sw:.28,sh:.28});ns.position.set(item.fw/2+.05,item.fh/2+.26,0);box.add(ns);var nc=item.cat==="essential"?"#00ffd0":item.cat==="violation"?"#ff9999":"#e2e8f0";var sfx=item.cat==="essential"?" \u2605":item.cat==="violation"?" \u26a0":"";var ls=mkSp(item.name+sfx,{fs:22,col:nc,sw:1.0,sh:.34,bg:"rgba(8,6,28,.9)"});ls.position.set(0,item.fh/2+.50,0);box.add(ls);var fl=mkSp("\u25b6 "+item.id+". "+item.name,{fs:30,col:"#00ffd0",sw:1.5,sh:.44,bg:"rgba(0,55,45,.96)"});fl.position.set(0,item.fh+.78,0);fl.visible=false;box.add(fl);bmap[item.id]={box:box,oMesh:oMesh,focusLabel:fl,mat:mat};});',
    'function setFocus(id){if(focId!==null){var p=bmap[focId];if(p){p.oMesh.material.opacity=0;p.focusLabel.visible=false;p.mat.opacity=p._orig||.88;}document.querySelectorAll(".ir.focused").forEach(function(r){r.classList.remove("focused");});}if(id===focId){focId=null;return;}focId=id;var e=bmap[id];if(!e)return;e.oMesh.material.opacity=.55;e.focusLabel.visible=true;e._orig=e.mat.opacity;e.mat.opacity=1.0;document.querySelectorAll(".ir").forEach(function(r){r.classList.remove("focused","active");if(parseInt(r.dataset.id)===id){r.classList.add("focused");r.scrollIntoView({block:"nearest",behavior:"smooth"});}});var item=mmap[e.box.uuid];if(item){var tx=item.px-RW/2,tz=item.pz,ox=panX,oz=panZ,s=0;function go(){s++;var t=s/30,ease=t<.5?2*t*t:(4-2*t)*t-1;panX=ox+(tx-ox)*ease;panZ=oz+(tz-oz)*ease;camUp();if(s<30)requestAnimationFrame(go);}requestAnimationFrame(go);}var p2=0;function pulse(){p2++;e.mat.opacity=1-Math.abs(Math.sin(p2*.25))*.35;if(p2<28&&focId===id)requestAnimationFrame(pulse);else e.mat.opacity=1.0;}requestAnimationFrame(pulse);}',
    'var list=document.getElementById("ilist");document.getElementById("sb-count").textContent=DATA.length+" items";',
    'DATA.forEach(function(item){var row=document.createElement("div");row.className="ir";row.dataset.id=item.id;var nBg=item.cat==="essential"?"#00ffd0":item.cat==="violation"?"#7a0d0d":"#3b2f8f";var nFc=item.cat==="essential"?"#000":"#fff";var dir=DIRMAP_[item.zone]||item.zone;var dc=DIRCOL_[item.zone]||"#888";row.innerHTML="<div class=\\"in\\" style=\\"background:"+nBg+";color:"+nFc+"\\">"+item.id+"</div><div class=\\"ie\\">"+item.emoji+"</div><div class=\\"ii\\"><div class=\\"iname\\">"+item.name+"</div><div class=\\"izone\\" style=\\"color:"+dc+"\\">"+dir+" "+(item.zone||"-")+"</div><div class=\\"istat "+item.cat+"\\">"+item.status+"</div></div>";row.addEventListener("click",function(){setFocus(item.id);});list.appendChild(row);});',
    'var tip=document.getElementById("tip");',
    'function showTip(item,ex,ey){tip.innerHTML="<span class=\\"te\\">"+item.emoji+"</span><div class=\\"tn\\">"+item.id+". "+item.name+"</div><div class=\\"tz\\">"+item.zone+"</div><div class=\\"ts "+item.cat+"\\">"+item.status+"</div>";tip.style.display="block";tip.style.left=(ex+12)+"px";tip.style.top=(ey+12)+"px";}',
    'function hideTip(){tip.style.display="none";}',
    'cvEl.addEventListener("mousemove",function(e){if(drag){hideTip();return;}mV.x=(e.clientX/W())*2-1;mV.y=-(e.clientY/H())*2+1;ray.setFromCamera(mV,camera);var hits=ray.intersectObjects(fGrp.children,true);var hit=null;for(var h=0;h<hits.length;h++){if(mmap[hits[h].object.uuid]){hit=hits[h];break;}}if(hit){var item=mmap[hit.object.uuid];showTip(item,e.clientX,e.clientY);document.querySelectorAll(".ir").forEach(function(r){if(parseInt(r.dataset.id)===item.id&&focId!==item.id)r.classList.add("active");else if(focId!==parseInt(r.dataset.id))r.classList.remove("active");});}else{hideTip();document.querySelectorAll(".ir:not(.focused)").forEach(function(r){r.classList.remove("active");});}});',
    'cvEl.addEventListener("click",function(e){mV.x=(e.clientX/W())*2-1;mV.y=-(e.clientY/H())*2+1;ray.setFromCamera(mV,camera);var hits=ray.intersectObjects(fGrp.children,true);for(var h=0;h<hits.length;h++){if(mmap[hits[h].object.uuid]){setFocus(mmap[hits[h].object.uuid].id);break;}}});',
    'buildFloor();buildCompass();',
    '(function loop(){requestAnimationFrame(loop);renderer.render(scene,camera);})();',
    'window.addEventListener("resize",function(){camera.aspect=W()/H();camera.updateProjectionMatrix();renderer.setSize(W(),H());});'
  ].join('\n');
}

function build3DHTML(data){
  var dataStr=JSON.stringify(data);
  var zcolsStr=JSON.stringify(ZCOLS);
  var zgridStr=JSON.stringify(ZGRID);
  var dirmapStr=JSON.stringify(DIRMAP);
  var dircolStr=JSON.stringify(DIRCOL);
  var script=viewer3DScript()
    .replace('__DATA__',dataStr)
    .replace('__ZCOLS__',zcolsStr)
    .replace('__ZGRID__',zgridStr)
    .replace('__DIRMAP__',dirmapStr)
    .replace('__DIRCOL__',dircolStr)
    .replace('__RW__',String(ROOM_W))
    .replace('__RD__',String(ROOM_D));
  var css=[
    '*{margin:0;padding:0;box-sizing:border-box;}',
    'body{background:#0d0d1a;font-family:Segoe UI,sans-serif;display:flex;height:100vh;overflow:hidden;}',
    '#cv{flex:1;position:relative;}',
    '#tip{position:absolute;display:none;pointer-events:none;background:rgba(8,6,28,.97);border:1.5px solid #7c6fe0;border-radius:12px;padding:10px 14px;color:#e8eaf6;font-size:12px;z-index:200;max-width:220px;}',
    '#tip .tn{font-size:13px;font-weight:800;color:#fff;margin-bottom:3px;}',
    '#tip .te{font-size:20px;float:right;margin-left:6px;}',
    '#tip .tz{color:#ffd966;font-size:10px;margin-bottom:2px;}',
    '#tip .ts{font-size:10px;font-weight:700;}',
    '#tip .ts.essential{color:#00ffd0;}#tip .ts.violation{color:#ff6666;}#tip .ts.detected{color:#88aacc;}',
    '#inst{position:absolute;bottom:8px;left:8px;z-index:50;background:rgba(8,6,28,.85);border:1px solid #3b2f8f;border-radius:7px;padding:5px 9px;color:#555;font-size:9px;line-height:1.7;}',
    '#sb{width:240px;flex-shrink:0;background:#0a0a1e;border-left:2px solid #3b2f8f;display:flex;flex-direction:column;overflow:hidden;}',
    '#sb-head{padding:8px 11px;background:#141430;border-bottom:1px solid #3b2f8f;font-size:11px;font-weight:700;color:#aac4ff;display:flex;align-items:center;gap:7px;}',
    '#sb-count{margin-left:auto;background:#3b2f8f;color:#aac4ff;border-radius:9px;padding:1px 6px;font-size:9px;}',
    '#cstrip{display:grid;grid-template-columns:1fr 1fr;gap:2px;padding:4px 6px;border-bottom:1px solid #3b2f8f;background:#0d0d22;}',
    '.cb{display:flex;align-items:center;justify-content:center;gap:2px;padding:3px 4px;border-radius:4px;font-size:9px;font-weight:800;}',
    '#ilist{flex:1;overflow-y:auto;padding:2px 0;}',
    '#ilist::-webkit-scrollbar{width:3px;}#ilist::-webkit-scrollbar-thumb{background:#3b2f8f;border-radius:2px;}',
    '.ir{display:flex;align-items:center;gap:6px;padding:6px 8px;cursor:pointer;border-left:3px solid transparent;transition:background .12s;position:relative;}',
    '.ir:hover{background:#131330;}.ir.active{background:#181840;border-left-color:#7c6fe0;}.ir.focused{background:#1a1a50;border-left-color:#00ffd0;}',
    '.ir.focused::after{content:"\\25BA";position:absolute;right:6px;color:#00ffd0;font-size:12px;font-weight:900;}',
    '.in{width:20px;height:20px;border-radius:50%;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:800;}',
    '.ie{font-size:15px;flex-shrink:0;}.ii{flex:1;min-width:0;}',
    '.iname{font-size:10px;font-weight:600;color:#e2e8f0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}',
    '.izone{font-size:8px;margin-top:1px;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}',
    '.istat{font-size:8px;font-weight:700;margin-top:1px;}',
    '.istat.essential{color:#00ffd0;}.istat.violation{color:#ff6666;}.istat.detected{color:#88aacc;}',
    '#sb-leg{padding:6px 9px;border-top:1px solid #3b2f8f;font-size:9px;color:#555;}',
    '.lr{display:flex;align-items:center;gap:5px;margin-bottom:2px;}',
    '.ls{width:11px;height:7px;border-radius:2px;flex-shrink:0;}'
  ].join('');
  var body='<div id="cv"><div id="tip"></div><div id="inst">Drag:rotate|Scroll:zoom|Right:pan<br>Click item to highlight</div></div>'
    +'<div id="sb"><div id="sb-head">Furniture<span id="sb-count">0</span></div>'
    +'<div id="cstrip">'
    +'<div class="cb" style="background:#0a1428;color:#4488ff">N</div>'
    +'<div class="cb" style="background:#280a0a;color:#ff4444">S</div>'
    +'<div class="cb" style="background:#201800;color:#ffaa00">W</div>'
    +'<div class="cb" style="background:#0a2010;color:#00cc55">E</div>'
    +'</div><div id="ilist"></div>'
    +'<div id="sb-leg">'
    +'<div class="lr"><div class="ls" style="background:#00b8a5;border:1px solid #00ffd0"></div>Essential</div>'
    +'<div class="lr"><div class="ls" style="background:#7a0d0d;border:1px solid #ff4444"></div>Violation</div>'
    +'<div class="lr"><div class="ls" style="background:#6688aa;border:1px solid #7788aa"></div>Detected</div>'
    +'</div></div>';
  return '<!DOCTYPE html><html><head><meta charset="UTF-8"/><style>'+css+'</style></head><body>'
    +body
    +'<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"><\/script>'
    +'<script>'+script+'<\/script>'
    +'</body></html>';
}

document.getElementById('iframe-cur').srcdoc=build3DHTML(CUR_DATA);
document.getElementById('iframe-opt').srcdoc=build3DHTML(OPT_DATA);
document.getElementById('cur-badge').textContent=CUR_DATA.length+' detected';
document.getElementById('opt-badge').textContent=OPT_DATA.length+' items';

function buildComparison(cmp){
  if(!cmp)return;
  var pos=(cmp.improvement||0)>=0;
  var ic=pos?'#00ffd0':'#ff4444';
  var ib=pos?'#0d2010':'#2a0a0a';
  var note=pos?'Compliance improved by '+cmp.improvement+'% after Vastu optimisation':'Review recommendations to improve compliance';
  document.getElementById('cmp-badge').textContent=(pos?'+':'')+cmp.improvement+'%';
  var sh='';
  var st=[['&#128269;','Detected by YOLO',cmp.detected_count,'#7c6fe0'],['&#128230;','Total (+ essential)',cmp.total_count,'#1565c0'],['&#9989;','Compliant zones',cmp.compliant_count,'#2e7d32'],['&#9888;','Zone violations',cmp.violation_count,'#b71c1c']];
  for(var s=0;s<st.length;s++){sh+='<div class="stat-card" style="border:1.5px solid '+st[s][3]+'44"><div class="stat-icon">'+st[s][0]+'</div><div class="stat-num" style="color:'+st[s][3]+'">'+st[s][2]+'</div><div class="stat-lbl">'+st[s][1]+'</div></div>';}
  document.getElementById('cmp-content').innerHTML=
    '<div class="cmp-title">&#128202; Vastu Compliance Comparison</div>'
    +'<div class="ratio-row">'
    +'<div class="ratio-card" style="background:#1e1e3a;border:1.5px solid #3b2f8f"><div class="ratio-label">Before Optimisation</div><div class="ratio-num" style="color:#e65100">'+cmp.compliance_before+'%</div><div class="ratio-sub">Vastu Compliant</div></div>'
    +'<div class="arrow" style="color:#3b2f8f">&rarr;</div>'
    +'<div class="ratio-card" style="background:#0d2010;border:1.5px solid #2e7d32"><div class="ratio-label">After Optimisation</div><div class="ratio-num" style="color:#00ffd0">'+cmp.compliance_after+'%</div><div class="ratio-sub">Vastu Compliant</div></div>'
    +'<div class="arrow" style="color:'+ic+'">'+(pos?'&#8593;':'&#8595;')+'</div>'
    +'<div class="ratio-card" style="background:'+ib+';border:1.5px solid '+ic+'"><div class="ratio-label">Improvement</div><div class="ratio-num" style="color:'+ic+'">'+(pos?'+':'')+cmp.improvement+'%</div><div class="ratio-sub">'+(pos?'Better compliance':'Needs review')+'</div></div>'
    +'</div>'
    +'<div class="stat-grid">'+sh+'</div>'
    +'<div class="prog-box">'
    +'<div class="prog-hdr"><span>Before: <b style="color:#e65100">'+cmp.compliance_before+'%</b></span><span>After: <b style="color:#00ffd0">'+cmp.compliance_after+'%</b></span></div>'
    +'<div class="prog-bar"><div class="prog-fill" style="width:'+cmp.compliance_before+'%;background:#e65100;opacity:.6"></div><div class="prog-fill" style="width:'+cmp.compliance_after+'%;background:linear-gradient(90deg,#2e7d32,#00ffd0)"></div></div>'
    +'<div class="prog-note" style="color:'+ic+'">'+note+'</div>'
    +'</div>';
}
"""
    w.write(JS)
    w.write('\n</script>\n')

    # Inject real comparison data
    # Use real stats if provided, otherwise zeros
    cmp_data = cmp_stats if cmp_stats else {
        'compliance_before': 0, 'compliance_after': 0,
        'improvement': 0, 'detected_count': n_cur,
        'total_count': n_opt, 'compliant_count': 0, 'violation_count': 0,
    }
    w.write('<script>buildComparison(' + json.dumps(cmp_data) + ');</script>\n')
    w.write('</body>\n</html>\n')

    html_content = w.getvalue()
    with open(fpath, 'w', encoding='utf-8') as fh:
        fh.write(html_content)
    logger.info('Combined interactive HTML saved -> ' + fpath)
    return fpath



def _generate_single_interactive_html(furn_list: list, room_dims: tuple,
                                       out_dir: str, tag: str,
                                       is_current: bool) -> str:
    """
    Generates a standalone interactive Three.js HTML for one layout.
    Returns the file path.
    """
    rw, rd, ch = room_dims
    n = len(furn_list)
    title_str = ('Current Layout — Detected Furniture'
                 if is_current else
                 'Vastu-Optimised Layout — All Furniture in Correct Zones')

    def _ser(lst):
        out = []
        for i, item in enumerate(lst):
            name   = item['object']
            is_ess = item.get('is_essential', False)
            bad    = item.get('action_needed', False)
            fw, fh, fd = FURNITURE.get(name, (0.8, 0.8, 0.8))
            c      = FURN_COLORS.get(name, [120, 140, 180, 255])
            zone_i = item.get('zone', '')
            raw    = item['position_3d']
            rx, _, rz = _remap_position(tuple(raw), rw, rd, zone=zone_i, index=i, total=n)
            px = float(rx + rw / 2)
            pz = float(rz)
            if bad:
                hex_col, opacity, cat = '#7a0d0d', 0.88, 'violation'
            elif is_ess:
                hex_col, opacity, cat = '#00b8a5', 0.92, 'essential'
            else:
                hex_col = '#{:02x}{:02x}{:02x}'.format(c[0], c[1], c[2])
                opacity, cat = 0.88, 'detected'
            compass = ZONE_COMPASS.get(zone_i, zone_i)
            emoji   = FURN_EMOJI.get(name, '\U0001f4e6')
            if is_current and bad:   s_txt = '\u26a0 Move needed'
            elif is_ess:             s_txt = '\u2605 Essential'
            elif not is_current:     s_txt = '\u2605 Vastu-correct'
            else:                    s_txt = '\u2713 Detected'
            out.append({'id': i+1, 'name': name, 'emoji': emoji,
                        'px': px, 'py': 0, 'pz': pz,
                        'fw': fw, 'fh': fh, 'fd': fd,
                        'color': hex_col, 'opacity': opacity, 'cat': cat,
                        'zone': zone_i, 'compass': compass, 'status': s_txt})
        return out

    data_js  = json.dumps(_ser(furn_list))
    zcols_js = json.dumps(ZONE_FLOOR_COLORS)
    badge    = 'OPTIMISED' if not is_current else 'CURRENT'
    badge_bg = '#006644'   if not is_current else '#3b2f8f'
    badge_fc = '#00ffd0'   if not is_current else '#aac4ff'

    html = _build_3d_html(
        title=title_str, data_js=data_js, zcols_js=zcols_js,
        rw=rw, rd=rd, ch=ch,
        badge=badge, badge_bg=badge_bg, badge_fc=badge_fc)

    fpath = os.path.join(out_dir, f'interactive_{tag}.html')
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(html)
    logger.info(f"Interactive {tag} HTML saved -> {fpath}")
    return fpath


def _build_3d_html(title, data_js, zcols_js, rw, rd, ch,
                   badge='', badge_bg='#3b2f8f', badge_fc='#aac4ff') -> str:
    """Shared Three.js HTML builder used by both single and combined views."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>{title}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:#0d0d1a;font-family:'Segoe UI',sans-serif;
     display:flex;height:100vh;overflow:hidden;}}
#cv{{flex:1;position:relative;min-width:0;}}
#topbar{{
  position:absolute;top:0;left:0;right:0;height:44px;
  background:rgba(13,13,26,.94);border-bottom:1px solid #3b2f8f;
  display:flex;align-items:center;padding:0 14px;gap:10px;z-index:50;
}}
#title-lbl{{color:#aac4ff;font-size:12px;font-weight:700;
            white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:1;}}
#mode-pill{{background:{badge_bg};color:{badge_fc};
  border-radius:10px;padding:3px 10px;font-size:10px;font-weight:700;white-space:nowrap;}}
#tip{{
  position:absolute;display:none;pointer-events:none;
  background:rgba(8,6,28,.97);border:1.5px solid #7c6fe0;
  border-radius:12px;padding:10px 14px;color:#e8eaf6;
  font-size:12px;z-index:200;max-width:240px;box-shadow:0 6px 24px #000b;
}}
#tip .tn{{font-size:14px;font-weight:800;color:#fff;margin-bottom:3px;}}
#tip .te{{font-size:22px;float:right;margin-left:8px;}}
#tip .tz{{color:#ffd966;font-size:11px;margin-bottom:1px;}}
#tip .td{{color:#64748b;font-size:10px;margin-bottom:3px;}}
#tip .ts{{font-size:11px;font-weight:700;}}
#tip .ts.essential{{color:#00ffd0;}}
#tip .ts.violation{{color:#ff6666;}}
#tip .ts.detected{{color:#88aacc;}}
#inst{{position:absolute;bottom:10px;left:10px;z-index:50;
  background:rgba(8,6,28,.85);border:1px solid #3b2f8f;
  border-radius:8px;padding:6px 10px;color:#555;font-size:10px;line-height:1.7;}}
#sb{{width:250px;flex-shrink:0;background:#0a0a1e;
  border-left:2px solid #3b2f8f;display:flex;flex-direction:column;overflow:hidden;}}
#sb-head{{padding:9px 12px;background:#141430;border-bottom:1px solid #3b2f8f;
  font-size:12px;font-weight:700;color:#aac4ff;display:flex;align-items:center;gap:8px;}}
#sb-count{{margin-left:auto;background:#3b2f8f;color:#aac4ff;
  border-radius:10px;padding:2px 7px;font-size:10px;}}
#cstrip{{display:grid;grid-template-columns:1fr 1fr;gap:3px;
  padding:5px 7px;border-bottom:1px solid #3b2f8f;background:#0d0d22;}}
.cbadge{{display:flex;align-items:center;justify-content:center;
  gap:3px;padding:3px 5px;border-radius:5px;font-size:10px;font-weight:800;}}
#ilist{{flex:1;overflow-y:auto;padding:3px 0;}}
#ilist::-webkit-scrollbar{{width:4px;}}
#ilist::-webkit-scrollbar-thumb{{background:#3b2f8f;border-radius:2px;}}
.irow{{display:flex;align-items:center;gap:7px;padding:7px 9px;cursor:pointer;
  border-left:3px solid transparent;transition:background .12s,border-color .12s;position:relative;}}
.irow:hover{{background:#131330;}}
.irow.active{{background:#181840;border-left-color:#7c6fe0;}}
.irow.focused{{background:#1a1a50;border-left-color:#00ffd0;box-shadow:inset 0 0 0 1px #00ffd022;}}
.irow.focused::after{{content:'→';position:absolute;right:7px;color:#00ffd0;font-size:13px;font-weight:900;}}
.inum{{width:22px;height:22px;border-radius:50%;flex-shrink:0;
  display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:800;}}
.iemoji{{font-size:16px;flex-shrink:0;}}
.iinfo{{flex:1;min-width:0;}}
.iname{{font-size:11px;font-weight:600;color:#e2e8f0;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
.izone{{font-size:9px;margin-top:1px;font-weight:700;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
.istat{{font-size:9px;font-weight:700;margin-top:1px;}}
.istat.essential{{color:#00ffd0;}}.istat.violation{{color:#ff6666;}}.istat.detected{{color:#88aacc;}}
#sb-leg{{padding:7px 10px;border-top:1px solid #3b2f8f;font-size:10px;color:#555;}}
.lrow{{display:flex;align-items:center;gap:6px;margin-bottom:3px;}}
.lsw{{width:12px;height:8px;border-radius:2px;flex-shrink:0;}}
</style>
</head>
<body>
<div id="cv">
  <div id="topbar">
    <div id="title-lbl">{title}</div>
    <div id="mode-pill">{badge}</div>
  </div>
  <div id="tip"></div>
  <div id="inst">&#x1f5b1; Drag: rotate &nbsp;|&nbsp; Scroll: zoom &nbsp;|&nbsp; Right-drag: pan<br>Click item to highlight in 3D</div>
</div>
<div id="sb">
  <div id="sb-head">&#x1f4cb; Furniture Index<span id="sb-count">0 items</span></div>
  <div id="cstrip">
    <div class="cbadge" style="background:#0a1428;color:#4488ff;border:1px solid #4488ff33">&#8593; N</div>
    <div class="cbadge" style="background:#280a0a;color:#ff4444;border:1px solid #ff444433">&#8595; S</div>
    <div class="cbadge" style="background:#201800;color:#ffaa00;border:1px solid #ffaa0033">&#8592; W</div>
    <div class="cbadge" style="background:#0a2010;color:#00cc55;border:1px solid #00cc5533">&#8594; E</div>
  </div>
  <div id="ilist"></div>
  <div id="sb-leg">
    <div class="lrow"><div class="lsw" style="background:#00b8a5;border:1px solid #00ffd0"></div><span>&#9733; Essential</span></div>
    <div class="lrow"><div class="lsw" style="background:#7a0d0d;border:1px solid #ff4444"></div><span>&#9888; Violation</span></div>
    <div class="lrow"><div class="lsw" style="background:#6688aa;border:1px solid #7788aa"></div><span>&#10003; Detected</span></div>
  </div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const ROOM_W={rw},ROOM_D={rd},ROOM_H={ch};
const ZCOLS={zcols_js};
const DATA={data_js};
const ZGRID=[['North-West',0,0,1/3,1/3],['North',1/3,0,1/3,1/3],['North-East',2/3,0,1/3,1/3],
  ['West',0,1/3,1/3,1/3],['Center',1/3,1/3,1/3,1/3],['East',2/3,1/3,1/3,1/3],
  ['South-West',0,2/3,1/3,1/3],['South',1/3,2/3,1/3,1/3],['South-East',2/3,2/3,1/3,1/3]];
const DIRMAP={{North:'↑N',South:'↓S',East:'→E',West:'←W',
  'North-East':'↗NE','North-West':'↖NW','South-East':'↘SE','South-West':'↙SW',Center:'⊕C'}};
const DIRCOL={{North:'#4488ff','North-East':'#4488ff','North-West':'#4488ff',
  South:'#ff4444','South-East':'#ff4444','South-West':'#ff4444',East:'#00cc55',West:'#ffaa00',Center:'#aaa'}};
const cvEl=document.getElementById('cv');
const W=()=>cvEl.clientWidth,H=()=>window.innerHeight;
const renderer=new THREE.WebGLRenderer({{antialias:true}});
renderer.setPixelRatio(window.devicePixelRatio);renderer.setSize(W(),H());renderer.shadowMap.enabled=true;
cvEl.appendChild(renderer.domElement);
const scene=new THREE.Scene();scene.background=new THREE.Color(0x0d0d1a);
scene.fog=new THREE.FogExp2(0x0d0d1a,.038);
const camera=new THREE.PerspectiveCamera(45,W()/H(),.1,120);
scene.add(new THREE.AmbientLight(0xffffff,.52));
const dl=new THREE.DirectionalLight(0xffffff,.88);dl.position.set(6,10,6);dl.castShadow=true;scene.add(dl);
scene.add(new THREE.HemisphereLight(0x334466,0x0d0d1a,.42));
let drag=false,rDrag=false,pm={{x:0,y:0}};
let theta=0.55,phi=1.05,rad=Math.sqrt(ROOM_W*ROOM_W+ROOM_D*ROOM_D)*1.9;
let panX=ROOM_W/2,panY=0,panZ=ROOM_D/2;
function camUp(){{camera.position.set(panX+rad*Math.sin(phi)*Math.sin(theta),panY+rad*Math.cos(phi),panZ+rad*Math.sin(phi)*Math.cos(theta));camera.lookAt(panX,panY,panZ);}}
camUp();
cvEl.addEventListener('mousedown',e=>{{drag=true;rDrag=e.button===2;pm={{x:e.clientX,y:e.clientY}};}});
cvEl.addEventListener('contextmenu',e=>e.preventDefault());
window.addEventListener('mouseup',()=>drag=false);
window.addEventListener('mousemove',e=>{{if(!drag)return;const dx=(e.clientX-pm.x)*.005,dy=(e.clientY-pm.y)*.005;if(rDrag){{const r=new THREE.Vector3();r.crossVectors(camera.getWorldDirection(new THREE.Vector3()),new THREE.Vector3(0,1,0)).normalize();panX-=r.x*dx*3;panZ-=r.z*dx*3;panY+=dy*3;}}else{{theta-=dx;phi=Math.max(.08,Math.min(Math.PI/2-.03,phi+dy));}}pm={{x:e.clientX,y:e.clientY}};camUp();}});
cvEl.addEventListener('wheel',e=>{{rad=Math.max(1.5,Math.min(40,rad+e.deltaY*.025));camUp();}});
let lt=null;cvEl.addEventListener('touchstart',e=>{{lt=e.touches[0];}},{{passive:true}});
cvEl.addEventListener('touchmove',e=>{{e.preventDefault();const t=e.touches[0];theta-=(t.clientX-lt.clientX)*.006;phi=Math.max(.08,Math.min(Math.PI/2-.03,phi+(t.clientY-lt.clientY)*.006));lt=t;camUp();}},{{passive:false}});
function mkSp(txt,o={{}}){{const fs=o.fs||48,col=o.col||'#fff',bg=o.bg||'rgba(0,0,0,0)';const cv=document.createElement('canvas');cv.width=256;cv.height=128;const x=cv.getContext('2d');x.fillStyle=bg;x.fillRect(0,0,256,128);x.font=`bold ${{fs}}px Segoe UI Emoji,Segoe UI,sans-serif`;x.fillStyle=col;x.textAlign='center';x.textBaseline='middle';x.fillText(txt,128,64);const sp=new THREE.Sprite(new THREE.SpriteMaterial({{map:new THREE.CanvasTexture(cv),transparent:true,depthTest:false}}));sp.scale.set(o.sw||.9,o.sh||.45,1);return sp;}}
function buildFloor(){{ZGRID.forEach(([zone,cx,cz,wf,df])=>{{const zw=wf*ROOM_W,zd=df*ROOM_D,x0=cx*ROOM_W,z0=cz*ROOM_D;const col=ZCOLS[zone]||'#ccc';const m=new THREE.Mesh(new THREE.PlaneGeometry(zw-.04,zd-.04),new THREE.MeshLambertMaterial({{color:new THREE.Color(col),transparent:true,opacity:.70,side:THREE.DoubleSide}}));m.rotation.x=-Math.PI/2;m.position.set(x0+zw/2,.001,z0+zd/2);scene.add(m);const el=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(zw,zd)),new THREE.LineBasicMaterial({{color:0xffffff,transparent:true,opacity:.16}}));el.rotation.x=-Math.PI/2;el.position.set(x0+zw/2,.002,z0+zd/2);scene.add(el);const sh=zone.replace('North','N').replace('South','S').replace('East','E').replace('West','W').replace('-','');const sp=mkSp(sh,{{fs:28,col:'#111',sw:.6,sh:.30}});sp.position.set(x0+zw/2,.04,z0+zd/2);scene.add(sp);}});}}
function buildCompass(){{[['N',ROOM_W/2,1.5,-.5,'#4488ff'],['S',ROOM_W/2,1.5,ROOM_D+.5,'#ff4444'],['W',-.5,1.5,ROOM_D/2,'#ffaa00'],['E',ROOM_W+.5,1.5,ROOM_D/2,'#00cc55']].forEach(([t,x,y,z,col])=>{{const cv=document.createElement('canvas');cv.width=130;cv.height=130;const ctx=cv.getContext('2d');ctx.fillStyle='rgba(8,6,28,.94)';ctx.beginPath();ctx.arc(65,65,62,0,Math.PI*2);ctx.fill();ctx.strokeStyle=col;ctx.lineWidth=5;ctx.beginPath();ctx.arc(65,65,60,0,Math.PI*2);ctx.stroke();ctx.font='bold 64px Segoe UI,sans-serif';ctx.fillStyle=col;ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(t,65,68);const sp=new THREE.Sprite(new THREE.SpriteMaterial({{map:new THREE.CanvasTexture(cv),transparent:true,depthTest:false}}));sp.scale.set(.85,.85,1);sp.position.set(x,y,z);scene.add(sp);}});[['↑ NORTH',ROOM_W/2,-.9,'#4488ff'],['SOUTH ↓',ROOM_W/2,ROOM_D+.9,'#ff4444'],['← WEST',-.9,ROOM_D/2,'#ffaa00'],['EAST →',ROOM_W+.9,ROOM_D/2,'#00cc55']].forEach(([t,x,z,col])=>{{const cv=document.createElement('canvas');cv.width=280;cv.height=76;const ctx=cv.getContext('2d');ctx.fillStyle='rgba(8,6,28,.9)';ctx.roundRect(0,0,280,76,12);ctx.fill();ctx.strokeStyle=col;ctx.lineWidth=2.5;ctx.roundRect(1.5,1.5,277,73,11);ctx.stroke();ctx.font='bold 38px Segoe UI Emoji,sans-serif';ctx.fillStyle=col;ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(t,140,38);const sp=new THREE.Sprite(new THREE.SpriteMaterial({{map:new THREE.CanvasTexture(cv),transparent:true,depthTest:false}}));sp.scale.set(1.75,.48,1);sp.position.set(x,.04,z);scene.add(sp);}});}}
const fGrp=new THREE.Group();scene.add(fGrp);
const ray=new THREE.Raycaster(),mV=new THREE.Vector2(-9,-9);
const mmap=new Map(),bmap=new Map();let focId=null;
DATA.forEach(item=>{{const hw=ROOM_W/2;const px=item.px-hw,py=item.fh/2,pz=item.pz;const geo=new THREE.BoxGeometry(item.fw,item.fh,item.fd);const mat=new THREE.MeshLambertMaterial({{color:new THREE.Color(item.color),transparent:true,opacity:item.opacity}});const box=new THREE.Mesh(geo,mat);box.position.set(px,py,pz);box.castShadow=true;fGrp.add(box);mmap.set(box.uuid,item);const ec=item.cat==='essential'?0x00ffd0:item.cat==='violation'?0xff4444:0x7788aa;box.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo),new THREE.LineBasicMaterial({{color:ec,linewidth:1.5}})));const oGeo=new THREE.BoxGeometry(item.fw*1.07,item.fh*1.07,item.fd*1.07);const oMat=new THREE.MeshBasicMaterial({{color:0x00ffd0,side:THREE.BackSide,transparent:true,opacity:0}});const oMesh=new THREE.Mesh(oGeo,oMat);box.add(oMesh);const es=mkSp(item.emoji,{{fs:46,sw:.56,sh:.56}});es.position.set(0,item.fh/2+.02,0);box.add(es);const nBg=item.cat==='essential'?'rgba(0,255,208,.95)':item.cat==='violation'?'rgba(180,20,20,.9)':'rgba(59,47,143,.9)';const nFc=item.cat==='essential'?'#000':'#fff';const ns=mkSp(String(item.id),{{fs:34,col:nFc,bg:nBg,sw:.30,sh:.30}});ns.position.set(item.fw/2+.05,item.fh/2+.28,0);box.add(ns);const nc=item.cat==='essential'?'#00ffd0':item.cat==='violation'?'#ff9999':'#e2e8f0';const sfx=item.cat==='essential'?' \u2605':item.cat==='violation'?' \u26a0':'';const ls=mkSp(item.name+sfx,{{fs:24,col:nc,sw:1.05,sh:.36,bg:'rgba(8,6,28,.9)'}});ls.position.set(0,item.fh/2+.52,0);box.add(ls);const fl=mkSp('\u25b6 '+item.id+'. '+item.name,{{fs:32,col:'#00ffd0',sw:1.55,sh:.46,bg:'rgba(0,55,45,.96)'}});fl.position.set(0,item.fh+.80,0);fl.visible=false;box.add(fl);bmap.set(item.id,{{box,oMesh,focusLabel:fl,mat}});}});
function setFocus(id){{if(focId!==null){{const p=bmap.get(focId);if(p){{p.oMesh.material.opacity=0;p.focusLabel.visible=false;p.mat.opacity=p._origOp||.88;}}document.querySelectorAll('.irow.focused').forEach(r=>r.classList.remove('focused'));}}if(id===focId){{focId=null;return;}}focId=id;const e=bmap.get(id);if(!e)return;e.oMesh.material.opacity=.55;e.focusLabel.visible=true;e._origOp=e.mat.opacity;e.mat.opacity=1.0;document.querySelectorAll('.irow').forEach(r=>{{r.classList.remove('focused','active');if(parseInt(r.dataset.id)===id){{r.classList.add('focused');r.scrollIntoView({{block:'nearest',behavior:'smooth'}});}}}});const item=mmap.get(e.box.uuid);if(item){{const tx=item.px-ROOM_W/2,tz=item.pz;const ox=panX,oz=panZ;let s=0;const go=()=>{{s++;const t=s/30;const ease=t<.5?2*t*t:(4-2*t)*t-1;panX=ox+(tx-ox)*ease;panZ=oz+(tz-oz)*ease;camUp();if(s<30)requestAnimationFrame(go);}};requestAnimationFrame(go);}}let p=0;const pulse=()=>{{p++;e.mat.opacity=1-Math.abs(Math.sin(p*.25))*.35;if(p<28&&focId===id)requestAnimationFrame(pulse);else e.mat.opacity=1.0;}};requestAnimationFrame(pulse);}}
const list=document.getElementById('ilist');
document.getElementById('sb-count').textContent=DATA.length+' items';
DATA.forEach(item=>{{const row=document.createElement('div');row.className='irow';row.dataset.id=item.id;const nBg=item.cat==='essential'?'#00ffd0':item.cat==='violation'?'#7a0d0d':'#3b2f8f';const nFc=item.cat==='essential'?'#000':'#fff';const dir=DIRMAP[item.zone]||item.zone;const dc=DIRCOL[item.zone]||'#888';row.innerHTML=`<div class="inum" style="background:${{nBg}};color:${{nFc}}">${{item.id}}</div><div class="iemoji">${{item.emoji}}</div><div class="iinfo"><div class="iname">${{item.name}}</div><div class="izone" style="color:${{dc}}">${{dir}} ${{item.zone||'—'}}</div><div class="istat ${{item.cat}}">${{item.status}}</div></div>`;row.addEventListener('click',()=>setFocus(item.id));list.appendChild(row);}});
const tip=document.getElementById('tip');
function showTip(item,ex,ey){{tip.innerHTML=`<span class="te">${{item.emoji}}</span><div class="tn">${{item.id}}. ${{item.name}}</div><div class="tz">&#128205; ${{item.zone}}</div><div class="td">&#129517; ${{item.compass}}</div><div class="ts ${{item.cat}}">${{item.status}}</div>`;tip.style.cssText+=`;display:block;left:${{ex+12}}px;top:${{ey+12}}px`;}}
function hideTip(){{tip.style.display='none';}}
cvEl.addEventListener('mousemove',e=>{{if(drag){{hideTip();return;}}mV.x=(e.clientX/W())*2-1;mV.y=-(e.clientY/H())*2+1;ray.setFromCamera(mV,camera);const hits=ray.intersectObjects(fGrp.children,true);const hit=hits.find(h=>mmap.has(h.object.uuid));if(hit){{const item=mmap.get(hit.object.uuid);showTip(item,e.clientX,e.clientY);document.querySelectorAll('.irow').forEach(r=>{{if(parseInt(r.dataset.id)===item.id&&focId!==item.id)r.classList.add('active');else if(focId!==parseInt(r.dataset.id))r.classList.remove('active');}});}}else{{hideTip();document.querySelectorAll('.irow:not(.focused)').forEach(r=>r.classList.remove('active'));}}}});
cvEl.addEventListener('click',e=>{{mV.x=(e.clientX/W())*2-1;mV.y=-(e.clientY/H())*2+1;ray.setFromCamera(mV,camera);const hits=ray.intersectObjects(fGrp.children,true);const hit=hits.find(h=>mmap.has(h.object.uuid));if(hit)setFocus(mmap.get(hit.object.uuid).id);}});
buildFloor();buildCompass();
(function loop(){{requestAnimationFrame(loop);renderer.render(scene,camera);}})();
window.addEventListener('resize',()=>{{camera.aspect=W()/H();camera.updateProjectionMatrix();renderer.setSize(W(),H());}});
</script></body></html>"""



def _generate_turntable_video(furn_list: list, room_dims: tuple,
                               out_dir: str, tag: str,
                               is_current: bool, fps: int = 24) -> str:
    """
    Render a 360° turntable animation of the 3D room and encode as MP4.
    Rotates the camera azimuth from 0→360° in 72 steps (every 5°),
    keeping elevation fixed at 28°. Saves as <tag>_turntable.mp4.
    Falls back to a GIF if cv2 VideoWriter fails.
    """
    import io
    rw, rd, ch = room_dims
    ess_set     = {i['object'] for i in furn_list if i.get('is_essential')}
    n_frames    = 72          # 72 × 5° = 360°
    elev        = 28
    title       = ('Current Layout — Detected Furniture'
                   if is_current else
                   '✨ Vastu-Optimised Layout')

    frames_rgb = []
    logger.info(f"Rendering {n_frames} frames for {tag} turntable…")

    for f_idx in range(n_frames):
        azim = f_idx * (360 / n_frames)

        fig = plt.figure(figsize=(12, 7), facecolor='#0d0d1a')
        ax  = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection='3d')

        _render_layout(ax, title, furn_list, room_dims, ess_set, is_current=is_current)
        ax.view_init(elev=elev, azim=azim)

        # Remove axis decorations for clean video
        ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')
        ax.tick_params(colors='#333333', labelsize=5)

        buf = io.BytesIO()
        plt.savefig(buf, dpi=100, bbox_inches='tight',
                    facecolor='#0d0d1a', format='png')
        plt.close(fig)
        buf.seek(0)

        img_arr = np.frombuffer(buf.read(), dtype=np.uint8)
        frame   = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if frame is not None:
            frames_rgb.append(frame)

        if (f_idx + 1) % 12 == 0:
            logger.info(f"  frame {f_idx+1}/{n_frames}")

    if not frames_rgb:
        logger.warning("No frames rendered for turntable video")
        return ''

    h, w = frames_rgb[0].shape[:2]
    out_path = os.path.join(out_dir, f'{tag}_turntable.mp4')

    # Try MP4 with H264, then XVID fallback, then GIF fallback
    written = False
    for fourcc_str in ('mp4v', 'XVID', 'MJPG'):
        try:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            ext    = '.mp4' if fourcc_str != 'MJPG' else '.avi'
            tmp    = os.path.join(out_dir, f'{tag}_turntable{ext}')
            vw     = cv2.VideoWriter(tmp, fourcc, fps, (w, h))
            if vw.isOpened():
                for fr in frames_rgb:
                    vw.write(fr)
                vw.release()
                out_path = tmp
                written  = True
                logger.info(f"Turntable video saved ({fourcc_str}) → {out_path}")
                break
            vw.release()
        except Exception as e:
            logger.warning(f"VideoWriter {fourcc_str} failed: {e}")

    if not written:
        # GIF fallback using matplotlib
        try:
            from PIL import Image as PILImg
            pil_frames = [PILImg.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                          for f in frames_rgb]
            out_path = os.path.join(out_dir, f'{tag}_turntable.gif')
            pil_frames[0].save(
                out_path, save_all=True,
                append_images=pil_frames[1:],
                duration=int(1000/fps), loop=0)
            logger.info(f"Turntable GIF saved → {out_path}")
        except Exception as e:
            logger.warning(f"GIF fallback also failed: {e}")
            return ''

    return out_path



def _generate_glb(furn_list: list, room_dims: tuple, out_path: str) -> bool:
    """
    Generate a GLB (GLTF Binary) file from the furniture layout.
    Uses manual GLTF2 JSON + binary buffer construction — no external library needed.
    Each furniture item becomes a colored box mesh at its position.
    Returns True if successful.
    """
    import struct, base64

    rw, rd, ch = room_dims
    n = len(furn_list)
    if n == 0:
        return False

    # Build geometry for each furniture box
    # GLTF2 format: positions + indices + materials
    all_positions = []   # flat list of x,y,z floats
    all_indices   = []   # flat list of uint16 triangle indices
    all_colors    = []   # per-mesh base color [r,g,b] 0-1
    all_names     = []
    vertex_offset = 0

    for i, item in enumerate(furn_list):
        name   = item['object']
        fw, fh, fd = FURNITURE.get(name, (0.8, 0.8, 0.8))
        c      = FURN_COLORS.get(name, [120, 140, 180, 255])
        zone_i = item.get('zone', '')
        raw    = item.get('position_3d', [0, 0, 0])
        rx, _, rz = _remap_position(tuple(raw), rw, rd,
                                     zone=zone_i, index=i, total=n)
        # Centre X on 0, Z from 0
        cx = float(rx)
        cy = float(fh / 2)
        cz = float(rz - rd / 2)  # shift so room centre = 0

        # 8 vertices of a box
        hw, hh, hd = fw/2, fh/2, fd/2
        verts = [
            (cx-hw, cy-hh, cz-hd), (cx+hw, cy-hh, cz-hd),
            (cx+hw, cy+hh, cz-hd), (cx-hw, cy+hh, cz-hd),
            (cx-hw, cy-hh, cz+hd), (cx+hw, cy-hh, cz+hd),
            (cx+hw, cy+hh, cz+hd), (cx-hw, cy+hh, cz+hd),
        ]
        all_positions.extend(verts)

        # 12 triangles (6 faces × 2 triangles)
        faces = [
            0,1,2, 0,2,3,  # front
            5,4,7, 5,7,6,  # back
            4,0,3, 4,3,7,  # left
            1,5,6, 1,6,2,  # right
            3,2,6, 3,6,7,  # top
            4,5,1, 4,1,0,  # bottom
        ]
        all_indices.extend([v + vertex_offset for v in faces])
        vertex_offset += 8

        col_r = c[0] / 255.0
        col_g = c[1] / 255.0
        col_b = c[2] / 255.0
        all_colors.append([col_r, col_g, col_b])
        all_names.append(name)

    # Pack binary buffer
    # Buffer view 0: positions (vec3 float32)
    pos_bytes = b''.join(
        struct.pack('<fff', x, y, z)
        for (x, y, z) in all_positions
    )
    # Buffer view 1: indices (uint16)
    idx_bytes = b''.join(struct.pack('<H', v) for v in all_indices)
    # Pad to 4-byte boundary
    if len(idx_bytes) % 4 != 0:
        idx_bytes += b'' * (4 - len(idx_bytes) % 4)

    bin_data   = pos_bytes + idx_bytes
    pos_offset = 0
    idx_offset = len(pos_bytes)
    n_verts    = len(all_positions)
    n_indices  = len(all_indices)

    # Compute bounding box for positions
    xs = [p[0] for p in all_positions]
    ys = [p[1] for p in all_positions]
    zs = [p[2] for p in all_positions]

    # Build GLTF JSON — one mesh per furniture item
    meshes = []
    mat_list = []
    nodes = []

    # We share position/index buffers but use accessors per mesh
    # Simpler: one mesh with all geometry, different material per primitive
    # For simplicity: one mesh total with all boxes as separate primitives
    accessors   = []
    buffer_views = [
        {"buffer": 0, "byteOffset": pos_offset,
         "byteLength": len(pos_bytes), "target": 34962},  # ARRAY_BUFFER
        {"buffer": 0, "byteOffset": idx_offset,
         "byteLength": len(idx_bytes), "target": 34963},  # ELEMENT_ARRAY_BUFFER
    ]

    # Position accessor
    accessors.append({
        "bufferView": 0,
        "componentType": 5126,  # FLOAT
        "count": n_verts,
        "type": "VEC3",
        "min": [min(xs), min(ys), min(zs)],
        "max": [max(xs), max(ys), max(zs)],
    })
    # Index accessor
    accessors.append({
        "bufferView": 1,
        "componentType": 5123,  # UNSIGNED_SHORT
        "count": n_indices,
        "type": "SCALAR",
    })

    # One material + primitive per furniture item
    primitives = []
    for i, (name, color) in enumerate(zip(all_names, all_colors)):
        mat_idx = len(mat_list)
        mat_list.append({
            "name": name,
            "pbrMetallicRoughness": {
                "baseColorFactor": [color[0], color[1], color[2], 1.0],
                "metallicFactor":  0.1,
                "roughnessFactor": 0.8,
            }
        })
        # Each primitive references same accessors but different index range
        prim_idx_start = i * 36  # 12 triangles × 3 verts
        accessors.append({
            "bufferView": 1,
            "byteOffset": prim_idx_start * 2,  # uint16 = 2 bytes
            "componentType": 5123,
            "count": 36,
            "type": "SCALAR",
        })
        accessors.append({
            "bufferView": 0,
            "byteOffset": i * 8 * 12,  # 8 verts × 12 bytes (vec3 float32)
            "componentType": 5126,
            "count": 8,
            "type": "VEC3",
        })
        primitives.append({
            "attributes": {"POSITION": len(accessors) - 1},
            "indices": len(accessors) - 2,
            "material": mat_idx,
        })

    meshes.append({"name": "Furniture", "primitives": primitives})
    nodes.append({"mesh": 0, "name": "Room"})

    gltf = {
        "asset": {"version": "2.0", "generator": "Vastu3D"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": mat_list,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(bin_data)}],
    }

    import json as _json
    json_bytes = _json.dumps(gltf, separators=(',', ':')).encode('utf-8')
    # Pad JSON to 4-byte boundary
    while len(json_bytes) % 4 != 0:
        json_bytes += b' '

    # GLB header: magic, version, total length
    magic   = b'glTF'
    version = struct.pack('<I', 2)
    json_chunk_len  = struct.pack('<I', len(json_bytes))
    json_chunk_type = struct.pack('<I', 0x4E4F534A)  # JSON
    bin_chunk_len   = struct.pack('<I', len(bin_data))
    bin_chunk_type  = struct.pack('<I', 0x004E4942)  # BIN

    total_len = 12 + 8 + len(json_bytes) + 8 + len(bin_data)
    total_len_bytes = struct.pack('<I', total_len)

    glb = (magic + version + total_len_bytes
           + json_chunk_len + json_chunk_type + json_bytes
           + bin_chunk_len  + bin_chunk_type  + bin_data)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        f.write(glb)
    logger.info(f"GLB saved -> {out_path} ({len(glb)} bytes)")
    return True


def _generate_renders(vastu_result: dict, video_path: str, out_dir: str, run_id: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    recs  = vastu_result.get('recommendations', [])
    rdims = (
        vastu_result['room_dimensions']['width_m'],
        vastu_result['room_dimensions']['depth_m'],
        vastu_result['room_dimensions']['height_m'],
    )
    ess_set = {r['object'] for r in recs if r.get('is_essential')}

    # ── LOG ALL PIPELINE KEYS (helps debug what key detections live under) ─
    logger.info(f"Pipeline result top-level keys: {list(vastu_result.keys())}")
    for key in ('detections', 'detected_objects', 'current_layout',
                'detected_furniture', 'recommendations'):
        val = vastu_result.get(key)
        if isinstance(val, list) and len(val) > 0:
            logger.info(f"  result['{key}'] has {len(val)} items: "
                        + str([i.get('object', i.get('name', '?'))
                                for i in val[:5]
                                if isinstance(i, dict)]))
        elif val is not None and not isinstance(val, list):
            logger.info(f"  result['{key}'] = {val!r} (not a list, skipping)")

    # ── DEBUG: print every rec's keys so we know the exact field names ──────
    logger.info(f"Total recs: {len(recs)}")
    for idx, r in enumerate(recs):
        logger.info(
            f"  rec[{idx}] keys={list(r.keys())} | "
            f"object={r.get('object')} | "
            f"is_essential={r.get('is_essential')} | "
            f"recommended_position_3d={r.get('recommended_position_3d')} | "
            f"recommended_zone={r.get('recommended_zone')} | "
            f"current_position_3d={r.get('current_position_3d')} | "
            f"current_zone={r.get('current_zone')}"
        )
        print(f"  [DEBUG rec {idx}] {r}")

    # ── BUILD CURRENT LIST (merged detections + recommendations) ──────────
    current_list = _build_current_list(vastu_result)

    # ── BUILD OPTIMISED LIST ──────────────────────────────────────────────
    # Optimised = ALL furniture at Vastu-recommended positions.
    # Detected items (is_essential=False) → moved to recommended zone
    # Essential items (is_essential=True) → placed in recommended zone
    # Both shown together — this is what the room SHOULD look like.
    opt_list = []
    for r in recs:
        if not r.get('object'):
            continue
        is_ess = bool(r.get('is_essential', False))
        opt_list.append({
            'object':                r.get('object', ''),
            'position_3d':           list(r.get('recommended_position_3d') or
                                           r.get('current_position_3d') or
                                           r.get('position_3d') or [0.0, 0.0, 0.0]),
            'zone':                  (r.get('recommended_zone') or
                                      r.get('current_zone') or
                                      r.get('zone', '')),
            'is_essential':          is_ess,
            'action_needed':         False,
            'not_detected_in_video': False,
        })
    print(f"[BUILD_OPT] opt_list ({len(opt_list)} items): "
          + ", ".join(f"{i['object']}(ess={i['is_essential']})" for i in opt_list))

    # Summary
    n_ghost   = sum(1 for i in current_list if _is_ghost(i, True))
    n_ess_det = sum(1 for i in current_list
                    if i.get('is_essential') and not _is_ghost(i, True))
    n_det     = sum(1 for i in current_list if not i.get('is_essential'))
    logger.info(
        f"current_list: {len(current_list)} items — "
        f"{n_ess_det} essential+detected (teal), "
        f"{n_ghost} ghost (essential, not in video), "
        f"{n_det} detected-only"
    )
    print(
        "[RENDER] current_list:\n" +
        "\n".join(f"  {i+1}. {item['object']} | zone={item['zone']} | "
                  f"ess={item.get('is_essential')} | "
                  f"not_detected={item.get('not_detected_in_video')} | "
                  f"→ {'GHOST' if _is_ghost(item, True) else 'SOLID'}"
                  for i, item in enumerate(current_list))
    )

    # ── UNet segmentation inset ───────────────────────────────────────────
    seg_inset = None
    try:
        from torchvision import transforms
        from PIL import Image as PILImage
        frames, _ = extract_frames_robust(
            video_path, sample_rate=FRAME_SAMPLE_RATE, max_frames=30)
        rgb   = frames[len(frames)//2]
        depth = estimate_depth(rgb, MIDAS_MODEL, MIDAS_TRANSFORM, DEVICE)
        dn    = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        unet  = _load_unet()
        if unet:
            img_t = transforms.Compose([
                transforms.Resize((256, 256)), transforms.ToTensor(),
                transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
            ])(PILImage.fromarray(rgb)).unsqueeze(0).to(DEVICE)
            d_u8  = (dn * 255).astype(np.uint8)
            dep_t = transforms.ToTensor()(
                PILImage.fromarray(d_u8).resize((256, 256))
            ).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred = unet(img_t, dep_t).argmax(dim=1).squeeze().cpu().numpy()
            pal = np.array([
                [70,130,180],[34,139,34],[205,133,63],[255,215,0],[178,34,34],
                [100,149,237],[255,165,0],[128,0,128],[220,220,220],[64,64,64],
            ], dtype=np.uint8)
            H, W = rgb.shape[:2]
            seg = cv2.addWeighted(rgb, 0.4,
                    cv2.resize(pal[pred % len(pal)], (W, H),
                               interpolation=cv2.INTER_NEAREST), 0.6, 0)
            seg_inset = seg
    except Exception as e:
        logger.warning(f"UNet inset skipped: {e}")

    paths = {}

    # ── Generate interactive HTML per layout (these ARE the 3D renders) ──
    # Each is a fully draggable/rotatable Three.js scene the user controls.
    for tag, furn_list, is_current in [
        ('current',   current_list, True),
        ('optimised', opt_list,     False),
    ]:
        # Interactive HTML (drag/rotate in browser)
        html_path = _generate_single_interactive_html(
            furn_list, rdims, out_dir, tag, is_current)
        paths[tag]            = html_path
        paths[f'{tag}_index'] = _generate_direction_index(furn_list, out_dir, tag)

        # Turntable video — generated in background thread (doesn't block response)
        paths[f'{tag}_video'] = ''  # will be populated by background thread
        import threading
        def _bg_turntable(fl=furn_list, rd_=rdims, od=out_dir,
                           tg=tag, ic=is_current):
            try:
                vp = _generate_turntable_video(fl, rd_, od, tg, ic, fps=20)
                logger.info(f"Background turntable done: {vp}")
            except Exception as ex:
                logger.warning(f"Background turntable failed: {ex}")
        threading.Thread(target=_bg_turntable, daemon=True).start()

    # Combined interactive (both layouts with toggle button)
    # Generate GLB files for download
    glb_dir = os.path.join(MEDIA_ROOT, "vastu_results", run_id, "3d_models")
    os.makedirs(glb_dir, exist_ok=True)
    cur_glb_path = os.path.join(glb_dir, "current_layout.glb")
    opt_glb_path = os.path.join(glb_dir, "optimised_layout.glb")
    try:
        _generate_glb(current_list, rdims, cur_glb_path)
        paths['current_glb'] = cur_glb_path
    except Exception as ex:
        logger.warning(f"Current GLB generation failed: {ex}")
        paths['current_glb'] = ''
    try:
        _generate_glb(opt_list, rdims, opt_glb_path)
        paths['optimised_glb'] = opt_glb_path
    except Exception as ex:
        logger.warning(f"Optimised GLB generation failed: {ex}")
        paths['optimised_glb'] = ''

    # Build real comparison stats and pass directly into the HTML generator
    recs_for_cmp = vastu_result.get('recommendations', [])
    real_cmp_stats = {
        'compliance_before': round(float(vastu_result.get('initial_compliance_pct', 0)), 1),
        'compliance_after':  round(float(vastu_result.get('final_compliance_pct', 0)), 1),
        'improvement':       round(float(vastu_result.get('final_compliance_pct', 0))
                                   - float(vastu_result.get('initial_compliance_pct', 0)), 1),
        'detected_count':    sum(1 for r in recs_for_cmp if not r.get('is_essential', False)),
        'total_count':       len(recs_for_cmp),
        'compliant_count':   sum(1 for r in recs_for_cmp if not r.get('action_needed', False)),
        'violation_count':   sum(1 for r in recs_for_cmp if r.get('action_needed', False)),
    }
    logger.info(f"Comparison stats: {real_cmp_stats}")

    paths['interactive_html'] = _generate_interactive_html(
        current_list, opt_list, rdims, out_dir, cmp_stats=real_cmp_stats)

    # ── Compliance chart ───────────────────────────────────────────────────
    total    = len(recs)
    n_viol_b = vastu_result.get('initial_violations', 0)
    n_viol_a = vastu_result.get('optimized_violations', 0)
    n_comp_b = total - n_viol_b
    n_comp_a = total - n_viol_a
    pct_b    = round(vastu_result.get('initial_compliance_pct', 0), 1)
    pct_a    = round(vastu_result.get('final_compliance_pct', 0), 1)

    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 6), facecolor='#12122a')
    fig2.suptitle('Vastu Compliance Analysis', fontsize=14, fontweight='bold',
                   color='white', y=1.02)
    for ax_p, comp, viol, pct, lbl in [
        (axes2[0], n_comp_b, n_viol_b, pct_b, f'Before\n{pct_b}% Compliant'),
        (axes2[1], n_comp_a, n_viol_a, pct_a, f'After Optimisation\n{pct_a}% Compliant'),
    ]:
        ax_p.set_facecolor('#12122a')
        ax_p.pie([comp, max(viol, 0)],
                 labels=[f'Compliant\n{comp}', f'Violations\n{viol}'],
                 colors=['#4caf50', '#f44336'], autopct='%1.0f%%', startangle=90,
                 textprops={'color': 'white', 'fontsize': 10},
                 wedgeprops=dict(width=0.55))
        ax_p.set_title(lbl, color='white', fontsize=11, fontweight='bold')

    ax_bar = axes2[2]; ax_bar.set_facecolor('#1e1e3a')
    names  = [r['object'][:12] for r in recs]
    scores = [r.get('vastu_score', 0) for r in recs]
    cols_b = ['#4caf50' if s >= 65 else '#ff9800' if s >= 45 else '#f44336' for s in scores]
    bars   = ax_bar.barh(names, scores, color=cols_b, edgecolor='#333', linewidth=0.5)
    ax_bar.set_xlim(0, 105)
    ax_bar.axvline(65, color='#ff9800', linestyle='--', linewidth=0.8, alpha=0.7)
    ax_bar.axvline(85, color='#4caf50', linestyle='--', linewidth=0.8, alpha=0.7)
    for bar, score in zip(bars, scores):
        ax_bar.text(score+1, bar.get_y()+bar.get_height()/2,
                    f'{score}', va='center', ha='left', fontsize=7, color='white')
    ax_bar.set_xlabel('Vastu Score', color='white', fontsize=9)
    ax_bar.set_title('Per-Furniture Score', color='white', fontsize=11, fontweight='bold')
    ax_bar.tick_params(colors='#aaa', labelsize=7)
    for sp in ['bottom', 'left']: ax_bar.spines[sp].set_color('#555')
    for sp in ['top', 'right']:   ax_bar.spines[sp].set_visible(False)

    plt.tight_layout()
    chart_path = os.path.join(out_dir, 'compliance_chart.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='#12122a')
    plt.close(fig2)
    logger.info(f"Compliance chart saved → {chart_path}")
    paths['chart'] = chart_path
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# VIEW — POST /run-full-pipeline/
# ─────────────────────────────────────────────────────────────────────────────

# ── Frontend key → pipeline name mapping ──────────────────────────────────────
FRONTEND_KEY_MAP = {
    # Bedroom
    "beds":         "bed",
    "wardrobe":     "wardrobe",
    "tables":       "desk",
    "tv":           "tv",
    "dustbin":      "dustbin",
    "chair":        "chair",
    "sofa":         "sofa",
    # Living Room
    "sofas":        "sofa",
    "painting":     "mirror",
    "plants":       "potted plant",
    "centerTable":  "dining table",
    # Kitchen
    "refrigerator": "refrigerator",
    "microwave":    "microwave",
    "sink":         "sink",
    "stove":        "oven",
    "diningTable":  "dining table",
    "coffeeMachine":"kettle",
    # Bathroom
    "toilet":       "toilet",
    "washbasin":    "sink",
    "mirror":       "mirror",
    "wasteBin":     "dustbin",
    "bathtub":      "bathtub",
    "tap":          "sink",
    # Study Room
    "studyTable":   "desk",
    "tableLamp":    "lamp",
    "computer":     "computer",
    "bookShelf":    "cabinet",
}

class RunFullPipelineAPIView(APIView):

    def post(self, request):
        temp_path = None
        try:
            video_file     = request.FILES.get('video')
            room_type      = request.data.get('room_type')
            furniture_data = request.data.get('furniture_data')

            if not video_file:
                return Response({'error': 'No video uploaded'},
                                status=status.HTTP_400_BAD_REQUEST)

            os.makedirs(os.path.join(MEDIA_ROOT, 'temp_videos'), exist_ok=True)
            filename  = f"{uuid.uuid4()}_{video_file.name}"
            temp_path = os.path.join(MEDIA_ROOT, 'temp_videos', filename)
            with open(temp_path, 'wb') as f:
                for chunk in video_file.chunks():
                    f.write(chunk)
            logger.info(f"Video saved: {temp_path}")

            try:
                raw_furniture = json.loads(furniture_data) if furniture_data else {}
            except Exception:
                raw_furniture = {}

            furniture_list = []
            if isinstance(raw_furniture, dict):
                for key, val in raw_furniture.items():
                    try:    count = int(val or 0)
                    except: count = 0
                    if count > 0:
                        mapped = FRONTEND_KEY_MAP.get(key, key)
                        if mapped not in furniture_list:
                            furniture_list.append(mapped)
            elif isinstance(raw_furniture, list):
                furniture_list = raw_furniture

            logger.info(f"User selected furniture: {furniture_list}")
            print("USER SELECTED FURNITURE:", furniture_list)

            load_shared_models()
            result = run_full_vastu_pipeline(
                user_video_path=temp_path,
                user_room_type=room_type,
                user_selected_furniture=furniture_list,
            )

            # Log all keys so we know where to find detections
            print("PIPELINE RESULT KEYS:", list(result.keys()))
            logger.info(f"Pipeline result keys: {list(result.keys())}")

            run_id = result.get("run_id")
            if not run_id:
                return Response({"error": "Pipeline did not return a run_id."},
                                status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            render_dir = os.path.join(MEDIA_ROOT, "vastu_results", run_id, "renders")
            os.makedirs(render_dir, exist_ok=True)
            
            render_paths = _generate_renders(result, temp_path, render_dir, run_id)

            logger.info(f"Renders: {render_paths}")

            # Read HTML file contents to embed as srcdoc (avoids iframe cross-origin block)
            def _read_html(path):
                try:
                    if path and os.path.exists(path):
                        with open(path, 'r', encoding='utf-8') as fh:
                            return fh.read()
                except Exception:
                    pass
                return ''

            # Comparison stats
            recs_list  = result.get('recommendations', [])
            n_detected = sum(1 for r in recs_list if not r.get('is_essential', False))
            n_total    = len(recs_list)
            n_viols    = sum(1 for r in recs_list if r.get('action_needed', False))
            n_compliant= n_total - n_viols
            pct_before = round(float(result.get('initial_compliance_pct', 0)), 1)
            pct_after  = round(float(result.get('final_compliance_pct', 0)), 1)

            result['3d_models'] = {
                # Direct URLs (open in new tab)
                "current_render":   _media_url(render_paths.get('current', '')),
                "optimised_render": _media_url(render_paths.get('optimised', '')),
                "compliance_chart": _media_url(render_paths.get('chart', '')),
                "current_index":    _media_url(render_paths.get('current_index', '')),
                "optimised_index":  _media_url(render_paths.get('optimised_index', '')),
                "interactive_html": _media_url(render_paths.get('interactive_html', '')),
                "current_glb":   _media_url(render_paths.get('current_glb', '')),
                "optimised_glb": _media_url(render_paths.get('optimised_glb', '')),
                # Turntable rotating videos
                "current_video":   _media_url(render_paths.get('current_video', '')),
                "optimised_video": _media_url(render_paths.get('optimised_video', '')),
                # HTML content embedded as srcdoc (no cross-origin issues)
                "current_html_content":   _read_html(render_paths.get('current', '')),
                "optimised_html_content": _read_html(render_paths.get('optimised', '')),
                # Comparison stats for UI display
                "comparison": {
                    "detected_count":    n_detected,
                    "total_count":       n_total,
                    "compliant_count":   n_compliant,
                    "violation_count":   n_viols,
                    "compliance_before": pct_before,
                    "compliance_after":  pct_after,
                    "improvement":       round(pct_after - pct_before, 1),
                },
            }

            return Response({"result": result, "video_url": _media_url(temp_path)})

        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"RunFullPipelineAPIView ERROR:\n{tb}")
            print("FULL TRACEBACK:\n", tb)
            return Response({"error": str(e), "traceback": tb},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)