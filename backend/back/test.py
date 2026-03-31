# test_3d_render.py
# ─────────────────────────────────────────────────────────────────────────────
# Run this script directly to test 3D room image generation from a video.
#
# Usage:
#   python test_3d_render.py --video "C:/path/to/your/room.mp4"
#
# Optional:
#   python test_3d_render.py --video "room.mp4" --pth "path/to/model.pth"
#
# Output:
#   test_output/current_3d_render.png
#   test_output/optimised_3d_render.png
#   test_output/depth_map.png
#   test_output/segmentation.png   (only if .pth is found)
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import argparse
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

print("=" * 60)
print("  3D Room Render — Standalone Test")
print("=" * 60)
VIDEO_PATH ="backend/media/uploads/room.mp4"

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--video', required=True,
                    help='Path to room video file (.mp4 / .avi / .mov etc.)')
parser.add_argument('--pth',   default=None,
                    help='Path to .pth model file (optional)')
parser.add_argument('--out',   default='test_output',
                    help='Output folder (default: test_output/)')
args = parser.parse_args()

VIDEO_PATH = args.video
PTH_PATH   = args.pth
OUT_DIR    = args.out
os.makedirs(OUT_DIR, exist_ok=True)

# Auto-find .pth if not given
if PTH_PATH is None:
    search_dirs = [
        'back/ml_models/MajorProject',
        'ml_models/MajorProject',
        'ml_models',
        '.',
    ]
    for d in search_dirs:
        if os.path.exists(d):
            for root, _, files in os.walk(d):
                for f in files:
                    if f.endswith('.pth'):
                        PTH_PATH = os.path.join(root, f)
                        break
            if PTH_PATH:
                break

print(f"\n📹 Video  : {VIDEO_PATH}")
print(f"🧠 Model  : {PTH_PATH or 'NOT FOUND — will skip segmentation'}")
print(f"📁 Output : {OUT_DIR}/")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"💻 Device : {DEVICE}\n")

# ── Vastu zone data ───────────────────────────────────────────────────────────
ZONE_FLOOR_COLORS = {
    'North-West': '#c3b1e1', 'North':  '#aec6cf', 'North-East': '#a8d8a8',
    'West':       '#f4a460', 'Center': '#eeeeee', 'East':       '#ffd966',
    'South-West': '#c8a882', 'South':  '#ff9999', 'South-East': '#ffb347',
}
ZONE_SHORT = {
    'North-East':'NE','North-West':'NW','South-East':'SE','South-West':'SW',
    'North':'N','South':'S','East':'E','West':'W','Center':'C',
}
ZONE_GRID_3D = [
    ('North-West',0,   0,   1/3,1/3), ('North',    1/3,0,   1/3,1/3),
    ('North-East',2/3, 0,   1/3,1/3), ('West',     0,  1/3, 1/3,1/3),
    ('Center',    1/3, 1/3, 1/3,1/3), ('East',     2/3,1/3, 1/3,1/3),
    ('South-West',0,   2/3, 1/3,1/3), ('South',    1/3,2/3, 1/3,1/3),
    ('South-East',2/3, 2/3, 1/3,1/3),
]
FURNITURE_SIZES = {
    'sofa':(2.0,0.8,0.9),'bed':(2.0,0.6,1.8),'tv':(1.2,0.7,0.15),
    'chair':(0.5,0.9,0.5),'desk':(1.4,0.75,0.7),'cabinet':(1.0,1.8,0.5),
    'dining table':(1.6,0.75,1.0),'refrigerator':(0.7,1.8,0.7),
    'toilet':(0.45,0.75,0.65),'sink':(0.6,0.2,0.5),
}
FURN_COLORS = {
    'sofa':[100,150,200],'bed':[150,100,100],'tv':[30,30,30],
    'chair':[139,90,60],'desk':[180,140,100],'cabinet':[140,110,80],
    'dining table':[160,120,80],'refrigerator':[220,220,220],
    'toilet':[255,255,255],'sink':[230,230,230],
}

def _hex_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2],16)/255 for i in (0,2,4))

def get_vastu_zone(cx, cy):
    v = 'North' if cy < 0.33 else ('Center' if cy < 0.66 else 'South')
    h = 'West'  if cx < 0.33 else ('Center' if cx < 0.66 else 'East')
    if v == 'Center' and h == 'Center': return 'Center'
    if v == 'Center': return h
    if h == 'Center': return v
    return v + '-' + h


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Extract middle frame from video
# ─────────────────────────────────────────────────────────────────────────────
print("─" * 50)
print("STEP 1: Extracting frames from video …")

if not os.path.exists(VIDEO_PATH):
    print(f"❌ ERROR: Video file not found: {VIDEO_PATH}")
    sys.exit(1)

cap   = cv2.VideoCapture(VIDEO_PATH)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps   = cap.get(cv2.CAP_PROP_FPS)
W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if total == 0 or W == 0:
    print("❌ ERROR: Cannot read video. Check the file format.")
    sys.exit(1)

print(f"   Resolution : {W}×{H}   |   FPS: {fps:.1f}   |   Frames: {total}")

# Extract up to 10 frames evenly spaced
step   = max(1, total // 10)
frames = []
fid    = 0
while True:
    ret, frame = cap.read()
    if not ret or len(frames) >= 10: break
    if fid % step == 0:
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    fid += 1
cap.release()

print(f"   ✅ Extracted {len(frames)} frames")
mid_frame = frames[len(frames) // 2]

# Save middle frame
cv2.imwrite(os.path.join(OUT_DIR, 'middle_frame.png'),
            cv2.cvtColor(mid_frame, cv2.COLOR_RGB2BGR))
print(f"   Saved: {OUT_DIR}/middle_frame.png")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — MiDaS depth estimation
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 50)
print("STEP 2: Running MiDaS depth estimation …")

try:
    midas = torch.hub.load('intel-isl/MiDaS', 'DPT_Hybrid', trust_repo=True)
    midas.to(DEVICE).eval()
    tfms  = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True)
    tfm   = tfms.dpt_transform
    print("   ✅ MiDaS DPT_Hybrid loaded")
except Exception as e:
    print(f"   ⚠ DPT_Hybrid failed ({e}), trying MiDaS_small …")
    midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', trust_repo=True)
    midas.to(DEVICE).eval()
    tfms  = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True)
    tfm   = tfms.small_transform

with torch.no_grad():
    inp  = tfm(mid_frame).to(DEVICE)
    pred = midas(inp)
    pred = F.interpolate(pred.unsqueeze(1), size=mid_frame.shape[:2],
                         mode='bicubic', align_corners=False).squeeze()
depth_map = np.nan_to_num(pred.cpu().numpy().astype(np.float32))
dn        = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)

# Save depth PNG
fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(dn, cmap='plasma')
plt.colorbar(im, ax=ax, label='Depth (near → far)')
ax.set_title('MiDaS Depth Map', fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'depth_map.png'), dpi=150)
plt.close()
print(f"   ✅ Depth map saved: {OUT_DIR}/depth_map.png")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — UNet segmentation (if .pth is available)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 50)
print("STEP 3: UNet segmentation …")

seg_overlay = None

if PTH_PATH and os.path.exists(PTH_PATH):
    print(f"   Loading .pth: {PTH_PATH}")

    class DoubleConv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch,out_ch,3,padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch,out_ch,3,padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            )
        def forward(self, x): return self.net(x)

    class UNet(nn.Module):
        def __init__(self, in_channels=4, num_classes=10):
            super().__init__()
            self.enc1=DoubleConv(in_channels,64); self.enc2=DoubleConv(64,128)
            self.enc3=DoubleConv(128,256);         self.enc4=DoubleConv(256,512)
            self.pool=nn.MaxPool2d(2)
            self.bottleneck=DoubleConv(512,1024)
            self.up4=nn.ConvTranspose2d(1024,512,2,stride=2); self.dec4=DoubleConv(1024,512)
            self.up3=nn.ConvTranspose2d(512,256,2,stride=2);  self.dec3=DoubleConv(512,256)
            self.up2=nn.ConvTranspose2d(256,128,2,stride=2);  self.dec2=DoubleConv(256,128)
            self.up1=nn.ConvTranspose2d(128,64,2,stride=2);   self.dec1=DoubleConv(128,64)
            self.final=nn.Conv2d(64,num_classes,1)
        def forward(self, img, depth):
            if depth.dim()==3: depth=depth.unsqueeze(1)
            depth=(depth-depth.min())/(depth.max()-depth.min()+1e-8)
            x=torch.cat([img,depth],dim=1)
            e1=self.enc1(x);       e2=self.enc2(self.pool(e1))
            e3=self.enc3(self.pool(e2)); e4=self.enc4(self.pool(e3))
            b=self.bottleneck(self.pool(e4))
            d4=self.dec4(torch.cat([self.up4(b),e4],dim=1))
            d3=self.dec3(torch.cat([self.up3(d4),e3],dim=1))
            d2=self.dec2(torch.cat([self.up2(d3),e2],dim=1))
            d1=self.dec1(torch.cat([self.up1(d2),e1],dim=1))
            return self.final(d1)

    try:
        from torchvision import transforms
        from PIL import Image as PILImage

        model = UNet(in_channels=4, num_classes=10)
        state = torch.load(PTH_PATH, map_location=DEVICE)
        if any(k.startswith('module.') for k in state):
            state = {k.replace('module.',''):v for k,v in state.items()}
        model.load_state_dict(state, strict=False)
        model.to(DEVICE).eval()
        print("   ✅ Model loaded successfully")

        img_t = transforms.Compose([
            transforms.Resize((256,256)), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])(PILImage.fromarray(mid_frame)).unsqueeze(0).to(DEVICE)

        d_u8  = (dn*255).astype(np.uint8)
        dep_t = transforms.ToTensor()(
            PILImage.fromarray(d_u8).resize((256,256))
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred_seg = model(img_t, dep_t).argmax(dim=1).squeeze().cpu().numpy()

        pal = np.array([
            [70,130,180],[34,139,34],[205,133,63],[255,215,0],[178,34,34],
            [100,149,237],[255,165,0],[128,0,128],[220,220,220],[64,64,64],
        ], dtype=np.uint8)

        seg_small  = pal[pred_seg % len(pal)]
        seg_full   = cv2.resize(seg_small, (W, H), interpolation=cv2.INTER_NEAREST)
        seg_overlay = cv2.addWeighted(mid_frame, 0.45, seg_full, 0.55, 0)

        # Save segmentation
        cv2.imwrite(os.path.join(OUT_DIR, 'segmentation.png'),
                    cv2.cvtColor(seg_overlay, cv2.COLOR_RGB2BGR))
        print(f"   ✅ Segmentation saved: {OUT_DIR}/segmentation.png")

    except Exception as e:
        print(f"   ❌ Model inference failed: {e}")
        print(f"      This means the .pth architecture may differ from UNet above.")
        seg_overlay = None
else:
    print("   ⚠ No .pth found — skipping segmentation")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Generate 3D room render with zone-coloured floor + furniture
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 50)
print("STEP 4: Generating 3D room renders …")

# Estimate room size from depth
room_w = 4.0
room_d = 3.5
ceil_h = 2.5

# Create synthetic furniture from detected depth positions
# (In real pipeline these come from vastu_3d_full.py)
# Here we simulate using the depth map to place test furniture
test_furniture = {
    'current': [
        {'object':'bed',    'cx':0.2, 'cy':0.75,'zone':'South-West','is_essential':True,  'action_needed':False},
        {'object':'tv',     'cx':0.8, 'cy':0.5, 'zone':'East',      'is_essential':False, 'action_needed':False},
        {'object':'chair',  'cx':0.5, 'cy':0.2, 'zone':'North',     'is_essential':False, 'action_needed':True},
        {'object':'desk',   'cx':0.8, 'cy':0.2, 'zone':'North-East','is_essential':True,  'action_needed':False},
        {'object':'cabinet','cx':0.1, 'cy':0.5, 'zone':'West',      'is_essential':False, 'action_needed':True},
    ],
    'optimised': [
        {'object':'bed',    'cx':0.2, 'cy':0.75,'zone':'South-West','is_essential':True,  'action_needed':False},
        {'object':'tv',     'cx':0.8, 'cy':0.4, 'zone':'East',      'is_essential':False, 'action_needed':False},
        {'object':'chair',  'cx':0.15,'cy':0.5, 'zone':'West',      'is_essential':False, 'action_needed':False},
        {'object':'desk',   'cx':0.8, 'cy':0.2, 'zone':'North-East','is_essential':True,  'action_needed':False},
        {'object':'cabinet','cx':0.15,'cy':0.75,'zone':'South-West','is_essential':False, 'action_needed':False},
    ],
}

def norm_to_3d(cx, cy, room_w, room_d, depth_map):
    """Convert normalised cx,cy → 3D X,Z position in room."""
    hw = room_w / 2
    X  = -hw + cx * room_w
    Z  = cy * room_d
    return X, 0.0, Z

def draw_zone_floor(ax, room_w, room_d):
    hw = room_w / 2
    for zone, cx, cz, wf, df in ZONE_GRID_3D:
        x0 = -hw + cx*room_w;  x1 = x0 + wf*room_w
        z0 = cz*room_d;         z1 = z0 + df*room_d
        col  = _hex_rgb(ZONE_FLOOR_COLORS.get(zone,'#eeeeee'))
        tile = [[[x0,0,z0],[x1,0,z0],[x1,0,z1],[x0,0,z1]]]
        ax.add_collection3d(Poly3DCollection(
            tile, alpha=0.55, facecolor=col,
            edgecolor='#888888', linewidth=0.6))
        mx,mz = (x0+x1)/2, (z0+z1)/2
        ax.text(mx, 0.02, mz, ZONE_SHORT.get(zone,zone),
                fontsize=7, ha='center', va='bottom', color='#222',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15',
                          facecolor='white', alpha=0.65, edgecolor='none'))

def draw_walls(ax, room_w, room_d, ceil_h):
    hw = room_w/2
    for face in [
        [[-hw,0,room_d],[hw,0,room_d],[hw,ceil_h,room_d],[-hw,ceil_h,room_d]],
        [[-hw,0,0],[-hw,0,room_d],[-hw,ceil_h,room_d],[-hw,ceil_h,0]],
        [[ hw,0,0],[ hw,0,room_d],[ hw,ceil_h,room_d],[ hw,ceil_h,0]],
    ]:
        ax.add_collection3d(Poly3DCollection(
            [face], alpha=0.07, facecolor=(0.78,0.72,0.9),
            edgecolor='#9575cd', linewidth=0.5))
    ax.text(0, 0.1, room_d+0.05, 'SOUTH',  fontsize=7, color='#ff6666', ha='center', fontweight='bold')
    ax.text(0, 0.1, -0.15,        'NORTH',  fontsize=7, color='#6699ff', ha='center', fontweight='bold')
    ax.text(-room_w/2-0.1,0.1,room_d/2,'WEST', fontsize=7, color='#cc8800', ha='right',  fontweight='bold')
    ax.text( room_w/2+0.1,0.1,room_d/2,'EAST', fontsize=7, color='#006633', ha='left',   fontweight='bold')

def draw_furniture_box(ax, name, pos3d, zone, is_essential, action_needed):
    X,_,Z = pos3d
    W2,H2,D2 = FURNITURE_SIZES.get(name,(0.8,0.8,0.8))
    x0,x1 = X-W2/2, X+W2/2
    y0,y1 = 0,        H2
    z0,z1 = Z-D2/2, Z+D2/2
    faces = [
        [[x0,y0,z0],[x1,y0,z0],[x1,y0,z1],[x0,y0,z1]],
        [[x0,y1,z0],[x1,y1,z0],[x1,y1,z1],[x0,y1,z1]],
        [[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0]],
        [[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]],
        [[x0,y0,z0],[x0,y0,z1],[x0,y1,z1],[x0,y1,z0]],
        [[x1,y0,z0],[x1,y0,z1],[x1,y1,z1],[x1,y1,z0]],
    ]
    col_list = FURN_COLORS.get(name,[160,160,160])
    r,g,b    = col_list[0]/255, col_list[1]/255, col_list[2]/255
    if is_essential:
        edge, lw = '#FFD700', 2.2
    elif action_needed:
        edge, lw = '#FF4444', 1.6
    else:
        edge, lw = '#444444', 0.6
    ax.add_collection3d(Poly3DCollection(
        faces, alpha=0.82, facecolor=(r,g,b),
        edgecolor=edge, linewidth=lw))
    prefix = '★ ' if is_essential else ''
    status = ' ⚠' if action_needed else ''
    label  = f"{prefix}{name}\n[{ZONE_SHORT.get(zone,zone)}]{status}"
    lbg    = '#1565c0' if is_essential else ('#ffcccc' if action_needed else 'white')
    lfc    = 'white'   if is_essential else ('#990000' if action_needed else '#111')
    ax.text(X, y1+0.08, Z, label,
            fontsize=6, ha='center', va='bottom', color=lfc,
            fontweight='bold' if is_essential else 'normal',
            linespacing=1.3,
            bbox=dict(boxstyle='round,pad=0.22',
                      facecolor=lbg, alpha=0.88, edgecolor='none'))

legend_patches = []
for zone, col in ZONE_FLOOR_COLORS.items():
    legend_patches.append(mpatches.Patch(
        facecolor=col, edgecolor='#888', linewidth=0.5,
        label=f"{ZONE_SHORT[zone]} = {zone}"))
legend_patches.append(mpatches.Patch(facecolor='#ccccff',edgecolor='#FFD700',linewidth=2, label='★ Essential'))
legend_patches.append(mpatches.Patch(facecolor='#ffcccc',edgecolor='#FF4444',linewidth=1.5,label='⚠ Violation'))

saved_paths = []

for layout_tag in ['current', 'optimised']:
    furn_list = test_furniture[layout_tag]
    title     = 'Current Layout' if layout_tag=='current' else '✨ Vastu-Optimised Layout'

    fig = plt.figure(figsize=(15, 9), facecolor='#12122a')
    ax  = fig.add_axes([0.0, 0.0, 0.72, 1.0], projection='3d')

    draw_zone_floor(ax, room_w, room_d)
    draw_walls(ax, room_w, room_d, ceil_h)

    for item in furn_list:
        pos3d = norm_to_3d(item['cx'], item['cy'], room_w, room_d, depth_map)
        draw_furniture_box(ax, item['object'], pos3d,
                           item['zone'], item['is_essential'], item['action_needed'])

    hw = room_w/2
    ax.set_xlim(-hw-0.5, hw+0.5)
    ax.set_ylim(-0.1,     ceil_h+0.5)
    ax.set_zlim(-0.5,     room_d+0.5)
    ax.set_xlabel('← West    East →',   fontsize=7, color='#cccccc', labelpad=4)
    ax.set_ylabel('Height (m)',           fontsize=7, color='#cccccc', labelpad=4)
    ax.set_zlabel('← North    South →',  fontsize=7, color='#cccccc', labelpad=4)
    ax.tick_params(colors='#aaaaaa', labelsize=6)
    ax.set_facecolor('#12122a')
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.set_facecolor('#12122a')
        pane.set_edgecolor('#3b2f8f')
        pane.fill = True
    ax.set_title(title, fontsize=12, fontweight='bold', color='white', pad=10)
    ax.view_init(elev=28, azim=40)

    # Legend panel
    ax_leg = fig.add_axes([0.73, 0.02, 0.26, 0.96])
    ax_leg.set_facecolor('#12122a'); ax_leg.axis('off')
    ax_leg.set_title('Zone & Status Key', color='white', fontsize=9, fontweight='bold', pad=4)
    ax_leg.legend(handles=legend_patches, loc='upper left',
                   fontsize=7, facecolor='#1e1e3a', labelcolor='white',
                   framealpha=0.9, edgecolor='#3b2f8f',
                   handlelength=1.4, handleheight=1.2)

    # UNet inset
    if seg_overlay is not None:
        ax_seg = fig.add_axes([0.74, 0.70, 0.24, 0.28])
        ax_seg.imshow(seg_overlay)
        ax_seg.set_title('UNet output', color='white', fontsize=7)
        ax_seg.axis('off')

    fname = os.path.join(OUT_DIR, f'{layout_tag}_3d_render.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight', facecolor='#12122a')
    plt.close(fig)
    saved_paths.append(fname)
    print(f"   ✅ Saved: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  TEST COMPLETE")
print("=" * 60)
print(f"\n📁 Output files in: {os.path.abspath(OUT_DIR)}/")
print()

all_files = [
    ('middle_frame.png',      'Middle frame extracted from video'),
    ('depth_map.png',         'MiDaS depth map'),
    ('segmentation.png',      'UNet segmentation overlay (.pth model)'),
    ('current_3d_render.png', '3D room — current furniture layout'),
    ('optimised_3d_render.png','3D room — vastu-optimised layout'),
]
for fname, desc in all_files:
    fpath = os.path.join(OUT_DIR, fname)
    if os.path.exists(fpath):
        size = os.path.getsize(fpath) // 1024
        print(f"  ✅  {fname:<28} ({size} KB)  — {desc}")
    else:
        print(f"  ⚠   {fname:<28}  NOT generated  — {desc}")

if seg_overlay is None:
    print("\n  ℹ  segmentation.png was not generated.")
    if PTH_PATH and os.path.exists(PTH_PATH):
        print("     The .pth was found but inference failed.")
        print("     → Check that your model architecture matches UNetSegmentation above.")
    else:
        print("     No .pth file was found. Use --pth to specify the path.")

print()
print("  To view the images, open the files above in any image viewer.")
print()