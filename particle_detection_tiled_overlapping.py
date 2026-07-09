"""
Tiled Particle Detection Gallery - FULL FEATURED
All features: summary table, gallery, full image zoom, individual edits,
mass edit, undo, sizing method display, bounding boxes

Features:
- Summary table (class × size bin)
- 6-column gallery with pagination
- Green bounding boxes on previews
- Sizing method display (edge_detect, mask_bounds, bbox)
- Full image zoom/pan with Plotly
- Individual class editing
- Delete individual particles
- Select + mass edit
- Undo stack
- CSV export
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
import pandas as pd
import json
import os
import io
import base64
import tempfile
import math
from datetime import datetime
from ultralytics import YOLO
from copy import deepcopy
import plotly.graph_objects as go
from scipy import ndimage
import torch

st.set_page_config(page_title="tiled dirt sniffer", page_icon="icon.ico", layout="wide")
st.title("🐕 tiled_dirt_sniffer: Review Dashboard")

# ============================================================
# Initialize Session State (FIRST!)
# ============================================================
if 'results' not in st.session_state:
    st.session_state.results = []
if 'tile_metadata' not in st.session_state:
    st.session_state.tile_metadata = []
if 'tile_files' not in st.session_state:
    st.session_state.tile_files = {}
if 'selected_particles' not in st.session_state:
    st.session_state.selected_particles = set()  # Keep for backwards compatibility, but not used
else:
    # Clean up any old string keys left from previous versions
    st.session_state.selected_particles = {p for p in st.session_state.selected_particles if isinstance(p, int)}
if 'undo_stack' not in st.session_state:
    st.session_state.undo_stack = []
if 'widget_counter' not in st.session_state:
    st.session_state.widget_counter = 0
if 'stitch_cache' not in st.session_state:
    st.session_state.stitch_cache = {}
if 'pipeline_stats' not in st.session_state:
    st.session_state.pipeline_stats = {}
if 'show_class_picker' not in st.session_state:
    st.session_state.show_class_picker = False
if 'notifications_enabled' not in st.session_state:
    st.session_state.notifications_enabled = False
if 'notif_shown' not in st.session_state:
    st.session_state.notif_shown = False


# Notification button - compact
with st.sidebar:
    col_sidebar1, col_sidebar2 = st.columns(2)
    with col_sidebar1:
        if st.button("🔔 Enable Notifications", key="notif_btn"):
            st.session_state.notifications_enabled = True
    with col_sidebar2:
        if st.button("↶ Undo", key="undo_sidebar"):
            if st.session_state.undo_stack:
                st.session_state.results = st.session_state.undo_stack.pop()
                st.success("✅ Undo!")
                st.rerun()
            else:
                st.warning("⚠️ Nothing to undo")

if st.session_state.notifications_enabled and not st.session_state.get('notif_shown'):
    st.info("✅ Notifications enabled!")
    st.session_state.notif_shown = True
    st.markdown("""
    <script>
    // Request notification permission
    if (Notification.permission === 'default') {
        Notification.requestPermission();
    }
    
    // Function to show notification
    window.showNotification = function(title, options) {
        if (Notification.permission === 'granted') {
            new Notification(title, {
                icon: '🐕',
                ...options
            });
        }
    };
    </script>
    """, unsafe_allow_html=True)



# ============================================================
# GPU/Device Detection - Support NVIDIA and Intel Arc
# ============================================================

def detect_device():
    """
    Auto-detect available GPU device
    YOLO only supports: CUDA (NVIDIA) or CPU
    Intel Arc XPU is detected but not directly supported by YOLO

    Priority: NVIDIA CUDA > CPU
    """
    device_info = {
        "device": None,
        "name": None,
        "memory_gb": 0,
        "backend": None
    }

    # Try NVIDIA CUDA
    if torch.cuda.is_available():
        try:
            device_count = torch.cuda.device_count()
            if device_count > 0:
                device = 0  # YOLO uses integer IDs for CUDA
                name = torch.cuda.get_device_name(0)
                memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                device_info["device"] = device
                device_info["name"] = name
                device_info["memory_gb"] = memory
                device_info["backend"] = "NVIDIA CUDA"
                return device_info
        except Exception as e:
            print(f"CUDA detection failed: {e}")

    # Intel Arc XPU - NOT directly supported by YOLO, fall back to CPU
    # (Note: YOLO only supports CUDA and CPU, not XPU)
    try:
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            print("⚠️ Intel Arc detected but YOLO doesn't support XPU directly")
            print("   Falling back to CPU. For Intel Arc, you need IPEX-optimized YOLO")
    except Exception as e:
        pass

    # Fall back to CPU
    device_info["device"] = 'cpu'
    device_info["name"] = "CPU"
    device_info["memory_gb"] = 0
    device_info["backend"] = "CPU"
    return device_info

# Detect device on startup
DEVICE_INFO = detect_device()
DEVICE = DEVICE_INFO["device"]

# CONFIG
MODEL_PATH = "models/best.pt"
CALIBRATION_UM_PER_PIXEL = 1.299

SIZE_BINS = [
    ("B: 5-15μm (1519 pcs)", 5, 15),
    ("C: 15-25μm (186 pcs)", 15, 25),
    ("D: 25-50μm (67 pcs)", 25, 50),
    ("E: 50-100μm (9 pcs)", 50, 100),
    ("F: 100-250μm (1 pcs)", 100, 250),
    ("G: 250-500μm (0 pcs)", 250, 500),
    ("H: 500-750μm (0 pcs)", 500, 750),
    ("I: 750-1000μm (0 pcs)", 750, 1000),
    ("J: 1000μm+ (0 pcs)", 1000, float("inf")),
]

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    model = YOLO(MODEL_PATH)
    # Note: YOLO handles device during inference via device parameter
    # Don't use .to() with YOLO models
    return model

def get_size_bin(diameter_um):
    """Get size bin label for a diameter in µm"""
    for label, lo, hi in SIZE_BINS:
        if lo <= diameter_um < hi:
            return label
    return SIZE_BINS[-1][0]

def simple_tile_dedup(raw_particles, tile_metadata):
    """
    Dedup pipeline based on metadata type:

    NO METADATA: Only over-labeling
    TILED: Over-labeling + edge deduplication + seam stitching
    STITCHED: Over-labeling + seam stitching
    """
    particles = [p.copy() for p in raw_particles]

    # Initialize deleted field for all particles
    for p in particles:
        if 'deleted' not in p:
            p['deleted'] = False

    # Determine metadata type
    if not tile_metadata:
        metadata_type = "NO_METADATA"
    else:
        # Check if tiles have neighbors (indicates tiled layout)
        has_neighbors = any(tile.get('neighbors') for tile in tile_metadata)
        metadata_type = "TILED" if has_neighbors else "STITCHED"

    st.write(f"**Metadata Type:** {metadata_type}")
    st.divider()

    # ─────────────────────────────────────────────────────────────────
    # STEP 1: Over-labeling (ALL metadata types)
    # ─────────────────────────────────────────────────────────────────
    st.write("**Step 1: Over-labeling detection (same spot, different class)**")

    overlabel_removed = 0

    def iou_2d(box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)
        if xi_max <= xi_min or yi_max <= yi_min:
            return 0.0
        inter = (xi_max - xi_min) * (yi_max - yi_min)
        union = (x1_max - x1_min) * (y1_max - y1_min) + (x2_max - x2_min) * (y2_max - y2_min) - inter
        return inter / union if union > 0 else 0.0

    for i, p1 in enumerate(particles):
        if p1.get('deleted'):
            continue

        for j, p2 in enumerate(particles[i+1:], start=i+1):
            if p2.get('deleted'):
                continue

            # Only same tile
            if p1.get('tile_filename') != p2.get('tile_filename'):
                continue

            # Different class (that's the point!)
            if p1.get('class') == p2.get('class'):
                continue

            # Same spot (within 20px)
            cx1 = p1['x'] + p1['w']/2
            cy1 = p1['y'] + p1['h']/2
            cx2 = p2['x'] + p2['w']/2
            cy2 = p2['y'] + p2['h']/2

            dist = ((cx1 - cx2)**2 + (cy1 - cy2)**2)**0.5

            if dist >= 20:
                continue

            # Check IOU and confidence
            iou = iou_2d((p1['x'], p1['y'], p1['x'] + p1['w'], p1['y'] + p1['h']),
                        (p2['x'], p2['y'], p2['x'] + p2['w'], p2['y'] + p2['h']))
            conf_diff = abs(p1['confidence'] - p2['confidence']) / max(p1['confidence'], p2['confidence'])

            # Delete if high overlap or confidence diff
            if iou > 0.7 or conf_diff > 0.3:
                if p1['confidence'] > p2['confidence']:
                    particles[j]['deleted'] = True
                    particles[j]['duplicate_type'] = 'location_duplicate'
                    particles[j]['duplicate_reason'] = f"Over-labeled: {p2['class']} vs {p1['class']}"
                    overlabel_removed += 1
                else:
                    particles[i]['deleted'] = True
                    particles[i]['duplicate_type'] = 'location_duplicate'
                    particles[i]['duplicate_reason'] = f"Over-labeled: {p1['class']} vs {p2['class']}"
                    overlabel_removed += 1
                    break

    st.write(f"  ✅ Removed {overlabel_removed} over-labeled particles")

    # Initialize stats
    dedup_stats = {
        'iou_removed': 0,
        'edge_duplicates_found': 0,
        'overlabel_removed': overlabel_removed,
        'stitches_found': 0
    }

    # If no metadata, we're done
    if metadata_type == "NO_METADATA":
        st.write("**No neighbor metadata available - skipping edge dedup and stitching**")
        return particles, dedup_stats

    # ─────────────────────────────────────────────────────────────────
    # STEP 2: Edge Deduplication (TILED ONLY)
    # ─────────────────────────────────────────────────────────────────
    if metadata_type == "TILED":
        st.write("**Step 2: Edge deduplication (particles at tile boundaries)**")

        # Build neighbor map
        neighbors_map = {}
        for tile in tile_metadata:
            fname = tile.get('filename')
            neighbors_map[fname] = tile.get('neighbors', {})

        edge_dup_count = 0
        edge_dup_deleted = 0
        edge_margin = 150
        score_threshold = 0.65

        for pidx, particle in enumerate(particles):
            if particle.get('deleted'):
                continue

            tile_file = particle.get('tile_filename')
            if not tile_file or tile_file not in neighbors_map:
                continue

            # Check if at edge
            x, y = particle.get('x', 0), particle.get('y', 0)
            w, h = particle.get('w', 0), particle.get('h', 0)
            cx, cy = x + w/2, y + h/2
            x_right = x + w
            y_bottom = y + h

            tile_w = particle.get('tile_width', 1024)
            tile_h = particle.get('tile_height', 1024)

            at_edges = []
            if cx < edge_margin or x < edge_margin:
                at_edges.append('left')
            if cx > tile_w - edge_margin or x_right > tile_w - edge_margin:
                at_edges.append('right')
            if cy < edge_margin or y < edge_margin:
                at_edges.append('top')
            if cy > tile_h - edge_margin or y_bottom > tile_h - edge_margin:
                at_edges.append('bottom')

            if not at_edges:
                continue

            # Score against neighbors
            for edge in at_edges:
                neighbor_file = neighbors_map[tile_file].get(edge)
                if not neighbor_file:
                    continue

                best_score = 0
                best_match = None
                best_match_conf = 0

                for nidx, neighbor in enumerate(particles):
                    if neighbor.get('deleted') or neighbor.get('tile_filename') != neighbor_file:
                        continue

                    # Class match
                    n_class = neighbor.get('class')
                    p_class = particle.get('class')
                    if p_class == n_class:
                        class_score = 1.0
                    elif (p_class in ['Fiber', 'Glass'] and n_class in ['Fiber', 'Glass']):
                        class_score = 0.7
                    elif (p_class in ['Metallic', 'Other'] and n_class in ['Metallic', 'Other']):
                        class_score = 0.7
                    else:
                        class_score = 0.3

                    # Size match
                    d1 = particle.get('diameter_um', 50)
                    d2 = neighbor.get('diameter_um', 50)
                    size_diff = abs(d1 - d2) / max(d1, d2, 1)
                    size_score = max(0, 1.0 - size_diff * 2.5)

                    # Position match
                    if edge in ['left', 'right']:
                        y_diff = abs(cy - (neighbor['y'] + neighbor['h']/2))
                        y_max = max(h, neighbor['h']) * 2.5
                        pos_score = max(0, 1.0 - (y_diff / y_max))
                    else:
                        x_diff = abs(cx - (neighbor['x'] + neighbor['w']/2))
                        x_max = max(w, neighbor['w']) * 2.5
                        pos_score = max(0, 1.0 - (x_diff / x_max))

                    score = (class_score * 0.25) + (size_score * 0.35) + (pos_score * 0.40)

                    if score > best_score:
                        best_score = score
                        best_match = nidx
                        best_match_conf = neighbor.get('confidence', 0)

                # Mark if good match and auto-delete lower confidence
                if best_score >= score_threshold and best_match is not None:
                    edge_dup_count += 1

                    current_conf = particle.get('confidence', 0)
                    if current_conf > best_match_conf:
                        particles[best_match]['deleted'] = True
                        particles[best_match]['duplicate_type'] = 'edge_duplicate'
                        particles[best_match]['duplicate_match'] = pidx
                        particles[best_match]['duplicate_score'] = round(best_score, 3)
                        particles[best_match]['matched_edge'] = edge
                        particles[best_match]['at_seam'] = True
                        edge_dup_deleted += 1
                    else:
                        particles[pidx]['deleted'] = True
                        particles[pidx]['duplicate_type'] = 'edge_duplicate'
                        particles[pidx]['duplicate_match'] = best_match
                        particles[pidx]['duplicate_score'] = round(best_score, 3)
                        particles[pidx]['matched_edge'] = edge
                        particles[pidx]['at_seam'] = True
                        edge_dup_deleted += 1
                    break

        st.write(f"  ✅ Found {edge_dup_count} edge duplicate pairs | 🗑️ Deleted {edge_dup_deleted}")
        dedup_stats['edge_duplicates_found'] = edge_dup_deleted

    # ─────────────────────────────────────────────────────────────────
    # STEP 3: Particle Stitching (TILED and STITCHED)
    # ─────────────────────────────────────────────────────────────────
    st.write(f"**Step {'3' if metadata_type == 'TILED' else '2'}: Detect particles for stitching (edge-touching)**")

    stitch_count = 0

    # Build neighbor map (for both TILED and STITCHED)
    neighbors_map = {}
    for tile in tile_metadata:
        fname = tile.get('filename')
        neighbors_map[fname] = tile.get('neighbors', {})

    # Find particles that touch edges
    for pidx, particle in enumerate(particles):
        if particle.get('deleted'):
            continue

        tile_file = particle.get('tile_filename')
        if not tile_file or tile_file not in neighbors_map:
            continue

        x, y = particle.get('x', 0), particle.get('y', 0)
        w, h = particle.get('w', 0), particle.get('h', 0)
        tile_w = particle.get('tile_width', 1024)
        tile_h = particle.get('tile_height', 1024)

        # Check if particle is near edge (150px margin - was working!)
        cx, cy = x + w/2, y + h/2
        x_right = x + w
        y_bottom = y + h

        edge_margin = 150  # This is what was working before!

        edges_near = []
        if cx < edge_margin or x < edge_margin:
            edges_near.append('left')
        if cx > tile_w - edge_margin or x_right > tile_w - edge_margin:
            edges_near.append('right')
        if cy < edge_margin or y < edge_margin:
            edges_near.append('top')
        if cy > tile_h - edge_margin or y_bottom > tile_h - edge_margin:
            edges_near.append('bottom')

        if not edges_near:
            continue

        # Score against neighbors
        for edge in edges_near:
            neighbor_file = neighbors_map[tile_file].get(edge)
            if not neighbor_file:
                continue

            best_score = 0
            best_match = None

            for nidx, neighbor in enumerate(particles):
                if neighbor.get('deleted') or neighbor.get('tile_filename') != neighbor_file:
                    continue

                # Size can be different (particles at seam might be different sizes)
                d1 = particle.get('diameter_um', 50)
                d2 = neighbor.get('diameter_um', 50)
                size_diff = abs(d1 - d2) / max(d1, d2, 1)
                if size_diff > 0.6:  # Very lenient: allows 60% difference
                    continue

                # Neighbor should be near opposite edge (more relaxed: 100px)
                n_x, n_y = neighbor.get('x', 0), neighbor.get('y', 0)
                n_w, n_h = neighbor.get('w', 0), neighbor.get('h', 0)
                n_tile_w = neighbor.get('tile_width', 1024)
                n_tile_h = neighbor.get('tile_height', 1024)

                opposite_edge_map = {'left': 'right', 'right': 'left', 'top': 'bottom', 'bottom': 'top'}
                opposite = opposite_edge_map.get(edge)

                neighbor_near_opposite = False
                if opposite == 'left' and n_x <= 100:  # More relaxed: was 50
                    neighbor_near_opposite = True
                elif opposite == 'right' and (n_x + n_w >= n_tile_w - 100):  # More relaxed: was 50
                    neighbor_near_opposite = True
                elif opposite == 'top' and n_y <= 100:  # More relaxed: was 50
                    neighbor_near_opposite = True
                elif opposite == 'bottom' and (n_y + n_h >= n_tile_h - 100):  # More relaxed: was 50
                    neighbor_near_opposite = True

                if not neighbor_near_opposite:
                    continue

                # Class match - be lenient (particles split at seams might have different labels)
                n_class = neighbor.get('class')
                p_class = particle.get('class')
                if p_class == n_class:
                    class_score = 1.0
                elif (p_class in ['Fiber', 'Glass'] and n_class in ['Fiber', 'Glass']):
                    class_score = 0.7
                elif (p_class in ['Metallic', 'Other'] and n_class in ['Metallic', 'Other']):
                    class_score = 0.7
                else:
                    class_score = 0.3

                # Size match - more lenient (50% tolerance)
                d1 = particle.get('diameter_um', 50)
                d2 = neighbor.get('diameter_um', 50)
                size_diff = abs(d1 - d2) / max(d1, d2, 1)
                size_score = max(0, 1.0 - size_diff * 2.0)  # More forgiving

                # Position check
                cx = x + w/2
                cy = y + h/2
                if edge in ['left', 'right']:
                    cy2 = n_y + n_h/2
                    y_diff = abs(cy - cy2)
                    max_h = max(h, n_h)
                    pos_score = max(0, 1.0 - (y_diff / (max_h * 2.5)))  # More lenient
                else:
                    cx2 = n_x + n_w/2
                    x_diff = abs(cx - cx2)
                    max_w = max(w, n_w)
                    pos_score = max(0, 1.0 - (x_diff / (max_w * 2.5)))  # More lenient

                # Combined score (class * 0.25 + size * 0.35 + position * 0.40)
                score = (class_score * 0.25) + (size_score * 0.35) + (pos_score * 0.40)

                if score > best_score and score > 0.35:  # Relaxed from 0.4 to 0.35
                    best_score = score
                    best_match = nidx

            # Mark for stitching
            if best_match is not None and best_score > 0.35:  # Relaxed from 0.4 to 0.35
                particles[pidx]['merged'] = True
                particles[pidx]['merge_type'] = 'stitched'
                particles[pidx]['matched_stitch'] = best_match
                particles[pidx]['stitch_edge'] = edge
                particles[pidx]['stitch_score'] = round(best_score, 3)
                stitch_count += 1
                break

    st.write(f"  ✅ Found {stitch_count} particles ready for stitching")
    dedup_stats['stitches_found'] = stitch_count

    return particles, dedup_stats
def calculate_particle_size_accurate(mask_array, calibration):
    """Edge detection sizing"""
    try:
        if mask_array is None or np.sum(mask_array) == 0:
            raise ValueError("Empty mask")
        edges = ndimage.sobel(mask_array.astype(float))
        edge_pixels = np.where(edges > 0.1)
        if len(edge_pixels[0]) > 0:
            y_min, y_max = edge_pixels[0].min(), edge_pixels[0].max()
            x_min, x_max = edge_pixels[1].min(), edge_pixels[1].max()
            diameter_pixels = max(x_max - x_min + 1, y_max - y_min + 1)
            return round(diameter_pixels * calibration, 1), "edge_detect"
    except:
        pass

    try:
        mask_pixels = np.where(mask_array > 0.5)
        if len(mask_pixels[0]) > 0:
            y_min, y_max = mask_pixels[0].min(), mask_pixels[0].max()
            x_min, x_max = mask_pixels[1].min(), mask_pixels[1].max()
            diameter_pixels = max(x_max - x_min + 1, y_max - y_min + 1)
            return round(diameter_pixels * calibration, 1), "mask_bounds"
    except:
        pass

    return None, "failed"

def calculate_merged_particle_size(stitched_image, calibration):
    """Recalculate size on the complete stitched image using edge detection"""

    try:
        if stitched_image is None or stitched_image.size == 0:
            return None, "failed"

        # Convert to grayscale for edge detection
        if len(stitched_image.shape) == 3:
            gray = cv2.cvtColor(stitched_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = stitched_image

        # Apply edge detection
        edges = ndimage.sobel(gray.astype(float))
        edge_pixels = np.where(edges > 0.1)

        if len(edge_pixels[0]) > 0:
            y_min, y_max = edge_pixels[0].min(), edge_pixels[0].max()
            x_min, x_max = edge_pixels[1].min(), edge_pixels[1].max()

            # True diameter from COMPLETE stitched particle
            diameter_pixels = max(x_max - x_min + 1, y_max - y_min + 1)
            diameter_um = diameter_pixels * calibration
            return round(diameter_um, 1), "merged_edge_detect"
    except:
        pass

    return None, "failed"

def stitch_merged_particle(tile_files, p, calibration=CALIBRATION_UM_PER_PIXEL):
    """Stitch together tiles for a merged cut particle and recalculate size"""

    if not p.get("merged"):
        return None, None, None

    try:
        # Get original particles that were merged
        originals = p.get("original_particles", [])
        if len(originals) < 2:
            return None, None, None

        # Load both tile images
        images = []
        for orig in originals:
            filename = orig["tile_filename"]
            if filename not in tile_files:
                return None, None, None

            file_obj = tile_files[filename]
            tile_img = Image.open(file_obj).convert('RGB')
            images.append(np.array(tile_img))

        if len(images) < 2:
            return None, None, None

        # Get positions of original particles
        img1, img2 = images[0], images[1]
        p1, p2 = originals[0], originals[1]

        # Simple stitch: side by side or top to bottom
        # Check which direction to stitch based on position
        seam_position = None
        if p1["tile_filename"] < p2["tile_filename"]:  # Rough ordering
            # Horizontal stitch (left-right)
            stitched = np.concatenate([img1, img2], axis=1)
            seam_position = {"type": "vertical", "pos": img1.shape[1]}  # Seam at x=width of img1
        else:
            # Vertical stitch (top-bottom)
            stitched = np.concatenate([img1, img2], axis=0)
            seam_position = {"type": "horizontal", "pos": img1.shape[0]}  # Seam at y=height of img1

        # RECALCULATE SIZE on complete stitched image
        merged_diameter_um, merged_method = calculate_merged_particle_size(stitched, calibration)

        return stitched, {
            "diameter_um": merged_diameter_um,
            "size_method": merged_method,
            "size_bin": get_size_bin(merged_diameter_um) if merged_diameter_um else "?"
        }, seam_position
    except:
        return None, None, None

def detect_particles_in_tiles(tile_files, tile_metadata, model):
    """Detect in all tiles (loads from uploaded files)"""
    all_particles = []
    progress_bar = st.progress(0)
    status = st.empty()

    # If no metadata, just iterate over uploaded files in order
    if not tile_metadata:
        file_list = list(tile_files.keys())
        for idx, filename in enumerate(file_list):
            status.text(f"Detecting {idx + 1}/{len(file_list)}: {filename}")

            try:
                file_obj = tile_files[filename]
                img_pil = Image.open(file_obj)
                if img_pil.mode != 'RGB':
                    img_pil = img_pil.convert('RGB')
                tile_img = np.array(img_pil)
            except Exception as e:
                st.warning(f"Failed to load {filename}: {e}")
                progress_bar.progress((idx + 1) / len(file_list))
                continue

            # Convert RGB to BGR for YOLO
            tile_img_bgr = cv2.cvtColor(tile_img, cv2.COLOR_RGB2BGR)

            # Detect
            try:
                results = model(tile_img_bgr, iou=0.45, conf=0.02, verbose=False, device=DEVICE)
            except Exception as e:
                st.warning(f"Detection failed on {filename}: {e}")
                progress_bar.progress((idx + 1) / len(file_list))
                continue

            # Extract particles
            for r in results:
                if r.boxes is None:
                    continue

                for i, (box, cls, conf) in enumerate(zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf)):
                    x1, y1, x2, y2 = [int(v) for v in box.tolist()]

                    # Get mask
                    try:
                        mask = r.masks.data[i].cpu().numpy() if hasattr(r.masks.data[i], 'cpu') else r.masks.data[i]
                    except:
                        mask = None

                    diameter_um, method = calculate_particle_size_accurate(mask, CALIBRATION_UM_PER_PIXEL)
                    if diameter_um is None:
                        diameter_um = 0

                    class_name = ["Fiber", "Glass", "Metallic", "Other"][int(cls)]

                    all_particles.append({
                        "tile_id": idx,
                        "tile_filename": filename,
                        "tile_width": tile_img.shape[1],
                        "tile_height": tile_img.shape[0],
                        "x": x1,
                        "y": y1,
                        "w": x2 - x1,
                        "h": y2 - y1,
                        "class": class_name,
                        "confidence": float(conf),
                        "diameter_um": diameter_um,
                        "size_method": method,
                        "size_bin": get_size_bin(diameter_um),
                    })

            progress_bar.progress((idx + 1) / len(file_list))

        progress_bar.empty()
        status.empty()
        return all_particles

    # Otherwise use metadata (tiled or stitched mode)
    for idx, tile_meta in enumerate(tile_metadata):
        filename = tile_meta.get('source_file') or tile_meta.get('filename')  # Support both field names
        if not filename:
            st.warning(f"Tile {idx} has no source_file or filename field")
            continue

        status.text(f"Detecting {idx + 1}/{len(tile_metadata)}: {filename}")

        # Load from uploaded file
        if filename not in tile_files:
            st.warning(f"Missing: {filename}")
            progress_bar.progress((idx + 1) / len(tile_metadata))
            continue

        try:
            file_obj = tile_files[filename]
            img_pil = Image.open(file_obj)
            if img_pil.mode != 'RGB':
                img_pil = img_pil.convert('RGB')
            tile_img = np.array(img_pil)
        except Exception as e:
            st.warning(f"Failed to load {filename}: {e}")
            progress_bar.progress((idx + 1) / len(tile_metadata))
            continue

        # Convert RGB to BGR for YOLO
        tile_img_bgr = cv2.cvtColor(tile_img, cv2.COLOR_RGB2BGR)

        # Detect
        try:
            results = model(tile_img_bgr, iou=0.45, conf=0.02, verbose=False, device=DEVICE)
        except Exception as e:
            st.warning(f"Detection failed on {filename}: {e}")
            progress_bar.progress((idx + 1) / len(tile_metadata))
            continue

        # Extract particles
        for r in results:
            if r.boxes is None:
                continue

            for i, (box, cls, conf) in enumerate(zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf)):
                x1, y1, x2, y2 = [int(v) for v in box.tolist()]

                # Get mask
                try:
                    mask = r.masks.data[i].cpu().numpy() if hasattr(r.masks.data[i], 'cpu') else r.masks.data[i]
                except:
                    mask = None

                diameter_um, method = calculate_particle_size_accurate(mask, CALIBRATION_UM_PER_PIXEL)
                if diameter_um is None:
                    diameter_um = max(x2 - x1, y2 - y1) * CALIBRATION_UM_PER_PIXEL
                    method = "bbox"

                all_particles.append({
                    "tile_id": idx,
                    "tile_filename": filename,
                    "tile_width": tile_img.shape[1],
                    "tile_height": tile_img.shape[0],
                    "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1,
                    "class": model.names[int(cls)],
                    "confidence": float(conf),
                    "diameter_um": diameter_um,
                    "size_bin": get_size_bin(diameter_um),
                    "size_method": method,
                    "deleted": False
                })

        progress_bar.progress((idx + 1) / len(tile_metadata))

    status.empty()
    return all_particles

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

if "results" not in st.session_state:
    st.session_state.results = None
if "undo_stack" not in st.session_state:
    st.session_state.undo_stack = []
if "stitch_cache" not in st.session_state:
    st.session_state.stitch_cache = {}
if "tile_metadata" not in st.session_state:
    st.session_state.tile_metadata = None
if "tile_files" not in st.session_state:
    st.session_state.tile_files = {}

def find_overlap_from_particles(tile_files, p1, p2, stitch_edge):
    """
    Find overlap by matching particle regions only
    Returns both overlap amount and match confidence score
    """
    try:
        file1 = tile_files.get(p1.get('tile_filename'))
        file2 = tile_files.get(p2.get('tile_filename'))

        if not file1 or not file2:
            return 50, 0.0

        img1 = cv2.cvtColor(np.array(Image.open(file1).convert('RGB')), cv2.COLOR_RGB2GRAY)
        img2 = cv2.cvtColor(np.array(Image.open(file2).convert('RGB')), cv2.COLOR_RGB2GRAY)

        padding = 80

        x1, y1, w1, h1 = p1.get('x', 0), p1.get('y', 0), p1.get('w', 50), p1.get('h', 50)
        x2, y2, w2, h2 = p2.get('x', 0), p2.get('y', 0), p2.get('w', 50), p2.get('h', 50)

        crop1_x1 = max(0, x1 - padding)
        crop1_y1 = max(0, y1 - padding)
        crop1_x2 = min(img1.shape[1], x1 + w1 + padding)
        crop1_y2 = min(img1.shape[0], y1 + h1 + padding)
        crop1 = img1[crop1_y1:crop1_y2, crop1_x1:crop1_x2]

        crop2_x1 = max(0, x2 - padding)
        crop2_y1 = max(0, y2 - padding)
        crop2_x2 = min(img2.shape[1], x2 + w2 + padding)
        crop2_y2 = min(img2.shape[0], y2 + h2 + padding)
        crop2 = img2[crop2_y1:crop2_y2, crop2_x1:crop2_x2]

        if crop1.size == 0 or crop2.size == 0:
            return 50, 0.0

        crop1_norm = crop1.astype(np.float32)
        crop2_norm = crop2.astype(np.float32)

        crop1_norm = (crop1_norm - crop1_norm.mean()) / (crop1_norm.std() + 1e-5)
        crop2_norm = (crop2_norm - crop2_norm.mean()) / (crop2_norm.std() + 1e-5)

        overlap_px = 50
        match_score = 0.0

        if stitch_edge in ['left', 'right']:
            edge_width = min(60, crop1.shape[1] // 2, crop2.shape[1] // 2)

            if edge_width < 10:
                return 50, 0.0

            crop1_right = crop1_norm[:, -edge_width:].astype(np.float32)
            crop2_left = crop2_norm[:, :edge_width].astype(np.float32)

            template_width = min(30, crop1_right.shape[1] // 2)
            if template_width < 5:
                return 50, 0.0

            template = crop1_right[:, -template_width:]

            if template.shape[0] <= 0 or template.shape[1] <= 0:
                return 50, 0.0
            if crop2_left.shape[0] < template.shape[0] or crop2_left.shape[1] < template.shape[1]:
                return 50, 0.0

            result = cv2.matchTemplate(crop2_left, template, cv2.TM_CCOEFF)
            if result.size > 0:
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                overlap_px = int(max_loc[0])
                match_score = float(max_val)  # Correlation coefficient
                overlap_px = max(10, min(overlap_px, 150))
        else:
            edge_height = min(60, crop1.shape[0] // 2, crop2.shape[0] // 2)

            if edge_height < 10:
                return 50, 0.0

            crop1_bottom = crop1_norm[-edge_height:, :].astype(np.float32)
            crop2_top = crop2_norm[:edge_height, :].astype(np.float32)

            template_height = min(30, crop1_bottom.shape[0] // 2)
            if template_height < 5:
                return 50, 0.0

            template = crop1_bottom[-template_height:, :]

            if template.shape[0] <= 0 or template.shape[1] <= 0:
                return 50, 0.0
            if crop2_top.shape[0] < template.shape[0] or crop2_top.shape[1] < template.shape[1]:
                return 50, 0.0

            result = cv2.matchTemplate(crop2_top, template, cv2.TM_CCOEFF)
            if result.size > 0:
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                overlap_px = int(max_loc[1])
                match_score = float(max_val)
                overlap_px = max(10, min(overlap_px, 150))

        return overlap_px, match_score

    except Exception as e:
        st.error(f"❌ Particle match error: {str(e)[:80]}")
        return 50, 0.0

def create_simple_stitched_view(tile_files, p1, p2, stitch_edge):
    """
    Stitch images using particle-region matching for accurate overlap detection

    Returns:
        stitched_img: aligned image with seam marked and boxes drawn
        merged_diameter_um: actual measured diameter from merged particle
    """
    try:
        # Load both tile images
        file1 = tile_files.get(p1.get('tile_filename'))
        file2 = tile_files.get(p2.get('tile_filename'))

        if not file1 or not file2:
            return None, 0

        img1 = Image.open(file1).convert('RGB')
        img2 = Image.open(file2).convert('RGB')

        img1_arr = np.array(img1)
        img2_arr = np.array(img2)

        # Reorder based on stitch edge
        if stitch_edge == 'left':
            img1_arr, img2_arr = img2_arr, img1_arr
            p1, p2 = p2, p1
        elif stitch_edge == 'top':
            img1_arr, img2_arr = img2_arr, img1_arr
            p1, p2 = p2, p1

        # Find overlap using particle regions
        overlap_px, match_score = find_overlap_from_particles(tile_files, p1, p2, stitch_edge)

        # Stitch full images with feather blending
        if stitch_edge in ['left', 'right']:
            new_width = img1_arr.shape[1] + img2_arr.shape[1] - overlap_px
            stitched = np.zeros((img1_arr.shape[0], new_width, 3), dtype=np.uint8)

            # Place first image
            stitched[:, :img1_arr.shape[1]] = img1_arr

            # Blend overlap region
            if overlap_px > 5:
                overlap_start = img1_arr.shape[1] - overlap_px
                for i in range(overlap_px):
                    alpha = i / overlap_px
                    stitched[:, overlap_start + i] = (
                        (1 - alpha) * img1_arr[:, img1_arr.shape[1] - overlap_px + i].astype(np.float32) +
                        alpha * img2_arr[:, i].astype(np.float32)
                    ).astype(np.uint8)

                stitched[:, overlap_start + overlap_px:] = img2_arr[:, overlap_px:]
            else:
                stitched[:, img1_arr.shape[1]:] = img2_arr
        else:
            new_height = img1_arr.shape[0] + img2_arr.shape[0] - overlap_px
            stitched = np.zeros((new_height, img1_arr.shape[1], 3), dtype=np.uint8)

            # Place first image
            stitched[:img1_arr.shape[0]] = img1_arr

            # Blend overlap region
            if overlap_px > 5:
                overlap_start = img1_arr.shape[0] - overlap_px
                for i in range(overlap_px):
                    alpha = i / overlap_px
                    stitched[overlap_start + i] = (
                        (1 - alpha) * img1_arr[img1_arr.shape[0] - overlap_px + i].astype(np.float32) +
                        alpha * img2_arr[i].astype(np.float32)
                    ).astype(np.uint8)

                stitched[overlap_start + overlap_px:] = img2_arr[overlap_px:]
            else:
                stitched[img1_arr.shape[0]:] = img2_arr

        # Draw boxes and seam
        stitched_pil = Image.fromarray(stitched)
        draw = ImageDraw.Draw(stitched_pil)

        x1, y1, w1, h1 = p1.get('x', 0), p1.get('y', 0), p1.get('w', 50), p1.get('h', 50)
        x2, y2, w2, h2 = p2.get('x', 0), p2.get('y', 0), p2.get('w', 50), p2.get('h', 50)

        if stitch_edge in ['left', 'right']:
            x1_offset = 0
            y1_offset = 0
            x2_offset = img1_arr.shape[1] - overlap_px if overlap_px > 0 else img1_arr.shape[1]
            y2_offset = 0

            draw.rectangle([(x1, y1), (x1 + w1, y1 + h1)], outline=(255, 165, 0), width=4)
            draw.rectangle([(x2 + x2_offset, y2), (x2 + x2_offset + w2, y2 + h2)], outline=(255, 255, 100), width=4)

            seam_x = img1_arr.shape[1] - overlap_px // 2
            stitched_arr = np.array(stitched_pil)
            if 0 <= seam_x < stitched_arr.shape[1]:
                stitched_arr[:, max(0, int(seam_x-4)):min(stitched_arr.shape[1], int(seam_x+4))] = [255, 0, 0]
            stitched_pil = Image.fromarray(stitched_arr)
        else:
            x1_offset = 0
            y1_offset = 0
            x2_offset = 0
            y2_offset = img1_arr.shape[0] - overlap_px if overlap_px > 0 else img1_arr.shape[0]

            draw.rectangle([(x1, y1), (x1 + w1, y1 + h1)], outline=(255, 165, 0), width=4)
            draw.rectangle([(x2, y2 + y2_offset), (x2 + w2, y2 + y2_offset + h2)], outline=(255, 255, 100), width=4)

            seam_y = img1_arr.shape[0] - overlap_px // 2
            stitched_arr = np.array(stitched_pil)
            if 0 <= seam_y < stitched_arr.shape[0]:
                stitched_arr[max(0, int(seam_y-4)):min(stitched_arr.shape[0], int(seam_y+4)), :] = [255, 0, 0]
            stitched_pil = Image.fromarray(stitched_arr)

        # Calculate merged size from particle positions
        x_min = min(x1 + x1_offset, x2 + x2_offset)
        y_min = min(y1 + y1_offset, y2 + y2_offset)
        x_max = max(x1 + x1_offset + w1, x2 + x2_offset + w2)
        y_max = max(y1 + y1_offset + h1, y2 + y2_offset + h2)

        merged_diameter_px = max(x_max - x_min, y_max - y_min)
        merged_diameter_um = merged_diameter_px * CALIBRATION_UM_PER_PIXEL

        return np.array(stitched_pil), merged_diameter_um

    except Exception as e:
        st.error(f"❌ Stitch error: {str(e)[:100]}")
        return None, 0
    """
    Fallback stitching if Stitcher fails
    Uses CLAHE + edge-based approach
    """
    try:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray1_clahe = clahe.apply(gray1)
        gray2_clahe = clahe.apply(gray2)

        edges1 = cv2.Canny(gray1_clahe, 50, 150)
        edges2 = cv2.Canny(gray2_clahe, 50, 150)

        overlap_px = 0
        if stitch_edge in ['left', 'right']:
            img1_right = edges1[:, -150:] if edges1.shape[1] >= 150 else edges1[:, -100:]
            img2_left = edges2[:, :150] if edges2.shape[1] >= 150 else edges2[:, :100]

            if img1_right.size > 0 and img2_left.size > 0:
                try:
                    result = cv2.matchTemplate(img2_left, img1_right[-50:, :], cv2.TM_CCOEFF)
                    if result.size > 0:
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                        overlap_px = int(max_loc[0])
                        overlap_px = max(0, min(overlap_px, 150))
                except:
                    overlap_px = 50
        else:
            img1_bottom = edges1[-150:, :] if edges1.shape[0] >= 150 else edges1[-100:, :]
            img2_top = edges2[:150, :] if edges2.shape[0] >= 150 else edges2[:100, :]

            if img1_bottom.size > 0 and img2_top.size > 0:
                try:
                    result = cv2.matchTemplate(img2_top, img1_bottom[-50:, :], cv2.TM_CCOEFF)
                    if result.size > 0:
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                        overlap_px = int(max_loc[1])
                        overlap_px = max(0, min(overlap_px, 150))
                except:
                    overlap_px = 50

        # Build stitched image
        if stitch_edge in ['left', 'right']:
            new_width = img1.shape[1] + img2.shape[1] - overlap_px
            stitched = np.zeros((img1.shape[0], new_width, 3), dtype=np.uint8)
            stitched[:, :img1.shape[1]] = img1

            if overlap_px > 5:
                overlap_start = img1.shape[1] - overlap_px
                for i in range(overlap_px):
                    alpha = i / overlap_px
                    stitched[:, overlap_start + i] = (
                        (1 - alpha) * img1[:, img1.shape[1] - overlap_px + i].astype(np.float32) +
                        alpha * img2[:, i].astype(np.float32)
                    ).astype(np.uint8)
                stitched[:, overlap_start + overlap_px:] = img2[:, overlap_px:]
            else:
                stitched[:, img1.shape[1]:] = img2
        else:
            new_height = img1.shape[0] + img2.shape[0] - overlap_px
            stitched = np.zeros((new_height, img1.shape[1], 3), dtype=np.uint8)
            stitched[:img1.shape[0]] = img1

            if overlap_px > 5:
                overlap_start = img1.shape[0] - overlap_px
                for i in range(overlap_px):
                    alpha = i / overlap_px
                    stitched[overlap_start + i] = (
                        (1 - alpha) * img1[img1.shape[0] - overlap_px + i].astype(np.float32) +
                        alpha * img2[i].astype(np.float32)
                    ).astype(np.uint8)
                stitched[overlap_start + overlap_px:] = img2[overlap_px:]
            else:
                stitched[img1.shape[0]:] = img2

        stitched_rgb = cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB)
        stitched_pil = Image.fromarray(stitched_rgb)
        draw = ImageDraw.Draw(stitched_pil)

        x1, y1, w1, h1 = p1.get('x', 0), p1.get('y', 0), p1.get('w', 50), p1.get('h', 50)
        x2, y2, w2, h2 = p2.get('x', 0), p2.get('y', 0), p2.get('w', 50), p2.get('h', 50)

        if stitch_edge in ['left', 'right']:
            x2_offset = img1.shape[1] - overlap_px
            y2_offset = 0
        else:
            x2_offset = 0
            y2_offset = img1.shape[0] - overlap_px

        draw.rectangle([(x1, y1), (x1 + w1, y1 + h1)], outline=(255, 165, 0), width=4)
        draw.rectangle([(x2 + x2_offset, y2 + y2_offset), (x2 + x2_offset + w2, y2 + y2_offset + h2)],
                      outline=(255, 255, 100), width=4)

        x_min = min(x1, x2 + x2_offset)
        y_min = min(y1, y2 + y2_offset)
        x_max = max(x1 + w1, x2 + x2_offset + w2)
        y_max = max(y1 + h1, y2 + y2_offset + h2)

        merged_diameter_px = max(x_max - x_min, y_max - y_min)
        merged_diameter_um = merged_diameter_px * CALIBRATION_UM_PER_PIXEL

        return np.array(stitched_pil), merged_diameter_um

    except:
        return None, 0


def push_undo():
    st.session_state.undo_stack.append(deepcopy(st.session_state.results))

def create_stitched_preview(tile_files, p1, p2, stitch_edge):
    """
    Create stitched preview - delegates to create_simple_stitched_view
    """
    return create_simple_stitched_view(tile_files, p1, p2, stitch_edge)

def display_particle_crop(pidx, p):
    """Display a single particle as a gallery thumbnail with ALL interactive features"""
    try:
        # Check if this is a merged particle - show stitched preview
        if p.get("merged") and p.get("matched_stitch") is not None:
            matched_idx = p.get("matched_stitch")

            # Only display the merged particle once (for the one with lower index)
            if pidx > matched_idx:
                return  # Skip this one, it's already shown with its partner

            if matched_idx < len(st.session_state.results):
                p2 = st.session_state.results[matched_idx]
                stitch_edge = p.get("stitch_edge")

                # Get the stitched image
                stitched_img, merged_diameter_um = create_simple_stitched_view(st.session_state.tile_files, p, p2, stitch_edge)

                if stitched_img is not None:
                    # Resize for gallery display
                    stitched_pil = Image.fromarray(stitched_img)
                    stitched_pil.thumbnail((400, 300), Image.Resampling.LANCZOS)
                    st.image(stitched_pil)

                    # Show merged info
                    st.caption(f"🔀 MERGED - {stitch_edge.upper()}\n{p.get('class', '?')} → {merged_diameter_um:.1f}µm")

                    # Generate widget ID for buttons
                    if "widget_counter" not in st.session_state:
                        st.session_state.widget_counter = 0
                    st.session_state.widget_counter += 1
                    widget_id = st.session_state.widget_counter

                    # EDIT CLASS - Change particle classification
                    current_class = p.get("class", "Other")
                    new_cls = st.selectbox(
                        "Class:",
                        ["Fiber", "Glass", "Metallic", "Other"],
                        index=["Fiber", "Glass", "Metallic", "Other"].index(current_class),
                        key=f"cls_{widget_id}",
                        label_visibility="collapsed"
                    )
                    if new_cls != current_class:
                        push_undo()
                        st.session_state.results[pidx]["class"] = new_cls
                        st.rerun()

                    # Stack buttons vertically for readability
                    if st.button("🔍 View", key=f"view_{widget_id}", use_container_width=True):
                        st.session_state[f"show_full_{pidx}"] = True
                        st.rerun()

                    if st.button("❌ Reject Match", key=f"reject_{pidx}", use_container_width=True):
                        push_undo()
                        matched_idx = p.get("matched_stitch")
                        # Unmerge both particles in the pair
                        st.session_state.results[pidx]["merged"] = False
                        st.session_state.results[pidx]["matched_stitch"] = None
                        # Also unmerge the matched particle
                        if matched_idx is not None and matched_idx < len(st.session_state.results):
                            st.session_state.results[matched_idx]["merged"] = False
                            st.session_state.results[matched_idx]["matched_stitch"] = None
                        st.success("✅ Match rejected! Both particles unmarked.")
                        st.rerun()

                    if st.button("🗑️ Delete", key=f"del_{widget_id}", use_container_width=True):
                        push_undo()
                        st.session_state.results[pidx]["deleted"] = True
                        st.rerun()

                return

        # Normal single particle display
        filename = p.get("tile_filename")
        if not filename or filename not in st.session_state.tile_files:
            st.warning("❌ Tile missing")
            return

        file_obj = st.session_state.tile_files[filename]
        tile_img = Image.open(file_obj).convert('RGB')
        tile_img_arr = np.array(tile_img)

        x, y, w, h = p.get("x", 0), p.get("y", 0), p.get("w", 10), p.get("h", 10)
        margin = 15
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(tile_img_arr.shape[1], x + w + margin)
        y2 = min(tile_img_arr.shape[0], y + h + margin)

        crop = tile_img_arr[y1:y2, x1:x2].copy()
        crop_pil = Image.fromarray(crop).convert('RGB')
        draw = ImageDraw.Draw(crop_pil)
        draw.rectangle([(x-x1, y-y1), (x+w-x1, y+h-y1)], outline=(0, 100, 255), width=2)

        aspect_ratio = crop_pil.height / crop_pil.width
        new_height = int(250 * aspect_ratio)
        crop_pil = crop_pil.resize((250, new_height), Image.Resampling.LANCZOS)

        st.image(crop_pil)

        # Caption with size bin and sizing method
        method = p.get("size_method", "?")
        caption = f"{p.get('class', '?')} | {p.get('size_bin', '?')}\n{p.get('diameter_um', '?'):.1f}µm ({method})\nConf: {p.get('confidence', 0):.2f}"

        # Show EDGE DUPLICATES (with score) - even if not deleted
        if p.get("is_duplicate") and p.get("duplicate_type") == "edge_duplicate":
            score = p.get("duplicate_score", 0)
            caption = f"🔗 EDGE DUPLICATE (Conf: {score})\n{caption}\n(Cut across seam)"
        # Mark deleted/duplicate particles (old types)
        elif p.get("deleted"):
            dup_type = p.get("duplicate_type", "unknown")
            if dup_type == "edge_duplicate":
                caption = f"❌ EDGE DUPLICATE\n{caption}\n🔪 Cut across tile boundary"
            elif dup_type == "location_duplicate":
                caption = f"❌ TWICE LABELED\n{caption}\n🏷️ Same spot, different class"
            else:
                caption = f"❌ DUPLICATE\n{caption}\n(Lower confidence)"
        # Mark merged particles
        elif p.get("merged"):
            caption = f"🔗 STITCHED\n{caption}\n{p.get('size_change_pct', 0):+.0f}%"

        # Add seam warning if applicable
        if p.get("at_seam") and not p.get("merged") and not p.get("is_duplicate"):
            caption += f"\n⚠️ At seams"

        st.caption(caption)

        # Show duplicate reason if applicable
        if p.get("deleted") and p.get("duplicate_reason"):
            st.info(f"**Why removed:** {p.get('duplicate_reason')}")

        # Generate widget ID for buttons
        if "widget_counter" not in st.session_state:
            st.session_state.widget_counter = 0
        st.session_state.widget_counter += 1
        widget_id = st.session_state.widget_counter

        # DIFFERENT BUTTONS BASED ON DELETED STATUS
        if not p.get("deleted"):
            # KEPT PARTICLES: Can edit, select, delete

            # CHECKBOX - Select particle for mass operations
            # Use session_state directly as checkbox state
            checkbox_key = f"chk_{pidx}"
            # Ensure the key exists in session state
            if checkbox_key not in st.session_state:
                st.session_state[checkbox_key] = False

            # Checkbox updates session_state directly
            st.checkbox("Select", value=st.session_state[checkbox_key], key=checkbox_key)

            # EDIT CLASS - Change particle classification (auto-save)
            current_class = p.get("class", "Other")
            new_cls = st.selectbox(
                "Class:",
                ["Fiber", "Glass", "Metallic", "Other"],
                index=["Fiber", "Glass", "Metallic", "Other"].index(current_class),
                key=f"cls_{widget_id}"
            )
            # Auto-save when class changes
            if new_cls != current_class:
                push_undo()
                st.session_state.results[pidx]["class"] = new_cls
                # Recalculate size bin for new class
                st.session_state.results[pidx]["size_bin"] = get_size_bin(st.session_state.results[pidx].get("diameter_um", 0))
                st.rerun()

            # DELETE BUTTON
            if st.button("🗑️ Delete", key=f"del_{widget_id}"):
                push_undo()
                st.session_state.results[pidx]["deleted"] = True
                st.rerun()
        else:
            # DELETED PARTICLES: Can only restore or view

            # RESTORE BUTTON - Bring particle back
            if st.button("♻️ Restore", key=f"restore_{widget_id}"):
                push_undo()
                st.session_state.results[pidx]["deleted"] = False
                # Clear deletion metadata
                st.session_state.results[pidx]["duplicate_reason"] = None
                st.session_state.results[pidx]["duplicate_type"] = None
                st.session_state.results[pidx]["matched_with"] = None
                st.rerun()

        # VIEW FULL BUTTON - Show for ALL particles (kept and deleted)
        if st.button("🔍 View", key=f"view_{widget_id}"):
            st.session_state[f"show_full_{pidx}"] = True
            st.rerun()

    except Exception as e:
        st.error(f"❌ Display error: {str(e)[:200]}")

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    # Show detected device at top
    st.markdown("---")
    st.write("**🖥️ Device Info:**")
    st.write(f"Backend: {DEVICE_INFO['backend']}")
    if DEVICE_INFO['name']:
        st.write(f"Device: {DEVICE_INFO['name']}")
    if DEVICE_INFO['memory_gb'] > 0:
        st.write(f"Memory: {DEVICE_INFO['memory_gb']:.1f} GB")
    st.markdown("---")

    st.header("📤 Upload Tiles")

    st.write("**Step 1: Upload manifest.json (optional)**")
    st.caption("For raw tiles with no metadata, leave this empty")
    manifest_file = st.file_uploader("Manifest:", type=["json"], key="manifest")

    st.write("**Step 2: Upload tile images**")
    tile_files = st.file_uploader(
        "Tile images:",
        type=["jpg", "jpeg", "png", "tif"],
        accept_multiple_files=True,
        key="tiles"
    )

    if tile_files and st.button("📋 Load"):
        try:
            file_map = {f.name: f for f in tile_files}

            if manifest_file:
                # Manifest provided - use metadata
                manifest = json.load(manifest_file)
                tile_metadata = manifest.get("tiles", [])
                st.session_state.tile_metadata = tile_metadata
                st.success(f"✅ Loaded {len(tile_metadata)} tiles with metadata")
            else:
                # No manifest - raw tiles mode
                tile_metadata = []
                st.session_state.tile_metadata = []
                st.success(f"✅ Loaded {len(file_map)} raw tile images (no metadata)")

            st.session_state.tile_files = file_map

        except Exception as e:
            st.error(f"Error: {e}")

    st.divider()

    if st.session_state.tile_files:
        if st.button("🔍 Run Inference"):
            model = load_model()
            if model is None:
                st.error("Model not found")
            else:
                # Show device being used
                device_emoji = "🟢" if DEVICE != 'cpu' else "🟡"
                st.info(f"{device_emoji} Running on: {DEVICE_INFO['backend']}")

                # Step 1: Detect in all tiles
                raw_particles = detect_particles_in_tiles(
                    st.session_state.tile_files,
                    st.session_state.tile_metadata,
                    model
                )
                st.write(f"**Raw detections: {len(raw_particles)}**")

                # Step 2: Simple tile dedup (IOU + neighbor scoring)
                st.divider()
                st.write("## 🔄 Processing")

                try:
                    final_particles, dedup_stats = simple_tile_dedup(
                        raw_particles,
                        st.session_state.tile_metadata
                    )

                    st.session_state.results = final_particles

                    # Calculate totals FIRST
                    # For merged particles: count as 1 (only count the lower index one)
                    total_active = 0
                    merged_count = 0
                    for i, p in enumerate(final_particles):
                        if not p.get("deleted"):
                            # If merged, only count if this is the lower index
                            if p.get("merged") and p.get("matched_stitch") is not None:
                                matched_idx = p.get("matched_stitch")
                                if i < matched_idx:  # Only count the lower index particle
                                    total_active += 1
                                    merged_count += 1
                            else:
                                # Non-merged particles count as 1
                                total_active += 1
                    total_edge_dups_deleted = len([p for p in final_particles if p.get("deleted") and p.get("duplicate_type") == "edge_duplicate"])
                    total_overlabeled = len([p for p in final_particles if p.get("deleted") and p.get("duplicate_type") == "location_duplicate"])
                    total_iou_deleted = len([p for p in final_particles if p.get("deleted") and p.get("duplicate_type") == "overlap_duplicate"])
                    total_deleted = total_edge_dups_deleted + total_overlabeled + total_iou_deleted

                    # Now store in pipeline stats
                    st.session_state.pipeline_stats = {
                        "raw_detections": len(raw_particles),
                        "iou_removed": dedup_stats['iou_removed'],
                        "overlabel_removed": dedup_stats['overlabel_removed'],
                        "edge_removed": dedup_stats['edge_duplicates_found'],
                        "merged_count": merged_count,
                        "total_active": total_active,
                        "total_deleted": total_deleted
                    }

                    # Summary
                    st.divider()

                    col_t1, col_t2, col_t3, col_t4, col_t5, col_t6 = st.columns(6)
                    col_t1.metric("Total Active", total_active)
                    col_t2.metric("Merged Pairs", merged_count)
                    col_t3.metric("Edge Dups Removed", total_edge_dups_deleted)
                    col_t4.metric("Over-Labeled Removed", total_overlabeled)
                    col_t5.metric("IOU Overlaps Removed", total_iou_deleted)
                    col_t6.metric("Total Deleted", total_deleted)

                    st.divider()

                    # Dedup process stats
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Raw Detections", len(raw_particles))
                    col2.metric("IOU Overlaps Removed", dedup_stats['iou_removed'])
                    col3.metric("Over-labeled Removed", dedup_stats['overlabel_removed'])
                    col4.metric("Edge Dups Removed", dedup_stats['edge_duplicates_found'])

                    st.success("✅ Processing complete!")

                    # Send notification if enabled
                    if st.session_state.notifications_enabled:
                        st.markdown("""
                        <script>
                        window.showNotification('🐕 Predictions Ready!', {
                            body: 'Your particle detection is complete. Check the results in the gallery.',
                            tag: 'detection-complete',
                            requireInteraction: true
                        });
                        </script>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Pipeline error: {e}")
                    import traceback
                    st.error(traceback.format_exc())
                    st.session_state.results = raw_particles

                st.session_state.undo_stack = []
                st.session_state.selected_particles = set()
                # Clear all checkbox states for new detection
                for idx in range(len(st.session_state.results)):
                    st.session_state[f"chk_{idx}"] = False
                st.success(f"✅ Done! Check galleries below.")
    if st.session_state.results:
        total = len([p for p in st.session_state.results if not p.get("deleted")])
        st.success(f"✅ {total} particles")
        # Count checked checkboxes
        selected_count = sum(1 for idx in range(len(st.session_state.results))
                            if st.session_state.get(f"chk_{idx}", False))
        st.write(f"**Selected:** {selected_count} (via checkboxes)")

    st.divider()

    if st.button("📥 Export CSV"):
        if st.session_state.results:
            rows = []
            for i, p in enumerate(st.session_state.results):
                if not p.get("deleted"):
                    # Skip merged particles where this is the higher index (will be exported with the lower index)
                    if p.get("merged") and p.get("matched_stitch") is not None:
                        matched_idx = p.get("matched_stitch")
                        if i > matched_idx:
                            continue  # Skip this one, it's already exported with the lower index

                    status = "MERGED (stitched)" if p.get("merged") else ("AT_SEAM (check)" if p.get("at_seam") else "OK")

                    # If merged, try to get recalculated size
                    diameter_um = p["diameter_um"]
                    size_method = p["size_method"]
                    size_bin = p["size_bin"]

                    if p.get("merged"):
                        try:
                            particle_key = f"{p.get('tile_filename')}_{p.get('x')}_{p.get('y')}"

                            if particle_key not in st.session_state.stitch_cache:
                                stitched, merged_meta, seam_info = stitch_merged_particle(st.session_state.tile_files, p)
                                if merged_meta:
                                    st.session_state.stitch_cache[particle_key] = merged_meta

                            if particle_key in st.session_state.stitch_cache:
                                merged_meta = st.session_state.stitch_cache[particle_key]
                                if merged_meta and merged_meta.get("diameter_um"):
                                    diameter_um = merged_meta["diameter_um"]
                                    size_method = merged_meta["size_method"]
                                    size_bin = merged_meta["size_bin"]
                                    status = f"MERGED_RECALC ({size_method})"
                        except Exception as e:
                            pass  # Keep original values if recalc fails

                    rows.append({
                        "tile": p["tile_filename"],
                        "class": p["class"],
                        "diameter_um": diameter_um,
                        "size_bin": size_bin,
                        "size_method": size_method,
                        "confidence": round(p["confidence"], 3),
                        "status": status,
                    })

            df = pd.DataFrame(rows)
            csv = df.to_csv(index=False)
            st.download_button(
                "⬇️ Download",
                csv,
                f"particles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.results is None:
    st.info("👈 Upload tiles and run inference")
else:
    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY TABLE
    # ─────────────────────────────────────────────────────────────────────────

    st.subheader("📊 Summary Table")

    data = {}
    for cls in ["Fiber", "Glass", "Metallic", "Other"]:
        data[cls] = {}
        for b, _, _ in SIZE_BINS:
            count = len([p for p in st.session_state.results
                        if p.get("class") == cls and p.get("size_bin") == b and not p.get("deleted")])
            data[cls][b] = count

    rows = []
    for cls in ["Fiber", "Glass", "Metallic", "Other"]:
        row = {"Material": cls}
        for b, _, _ in SIZE_BINS:
            c = data[cls][b]
            row[b] = c
        rows.append(row)

    # Add totals row
    totals_row = {"Material": "TOTAL"}
    for b, _, _ in SIZE_BINS:
        total = sum(data[cls][b] for cls in ["Fiber", "Glass", "Metallic", "Other"])
        totals_row[b] = total
    rows.append(totals_row)

    df = pd.DataFrame(rows)
    st.dataframe(df, width='stretch', height=200)

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # GALLERY
    # ─────────────────────────────────────────────────────────────────────────

    st.subheader("🖼️ Particle Gallery")

    # Show pipeline breakdown in dropdown
    if st.session_state.get('pipeline_stats'):
        stats = st.session_state.pipeline_stats
        with st.expander("📊 Pipeline Breakdown", expanded=False):
            st.markdown(f"""
            **Raw Detections:** {stats['raw_detections']} 🔍
            
            ↓ **Processing Steps:**
            
            - **IOU Overlaps Removed:** {stats['iou_removed']} ❌
            - **Over-labeled Removed:** {stats['overlabel_removed']} ❌
            - **Edge Dups Removed:** {stats['edge_removed']} ❌
            - **Particles Stitched/Merged:** {stats.get('merged_count', 0)} 🔀 (counted as 1 particle)
            
            ↓ **Final Results:**
            
            - **Total Active Particles:** {stats['total_active']} ✅ (merged pairs = 1 count)
            - **Total Deleted:** {stats['total_deleted']} (sum of all removals)
            """)

    st.divider()

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        filter_class = st.multiselect(
            "Class:",
            ["Fiber", "Glass", "Metallic", "Other"],
            default=["Fiber", "Glass", "Metallic", "Other"],
            key="fc"
        )
    with col2:
        filter_bins = st.multiselect(
            "Size Bin:",
            [b[0] for b in SIZE_BINS],
            default=[b[0] for b in SIZE_BINS],
            key="fb"
        )
    with col3:
        show_seams_only = st.checkbox("Seams only")
    with col4:
        show_merged_only = st.checkbox("Merged only")
    with col5:
        sort_by = st.selectbox("Sort:", ["Confidence ↓", "Confidence ↑", "Size ↓", "Size ↑"], index=0)
    with col6:
        items_per_page = st.selectbox("Per page:", [12, 18, 24, 36], index=0)

    # Show/Hide particle types
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        show_edge_dups = st.checkbox("🔗 Edge Duplicates (only)", value=False)
    with col_s2:
        show_overlabeled = st.checkbox("🏷️ Over-Labeled (only)", value=False)
    with col_s3:
        show_merged = st.checkbox("🔀 Merged/Stitched (only)", value=False)
    with col_s4:
        show_deleted = st.checkbox("❌ Deleted (only) - All", value=False)

    # Filter particles
    all_particles = []
    if st.session_state.results is not None:
        # Debug: count particles by status
        total = len(st.session_state.results)
        deleted_count = len([p for p in st.session_state.results if p.get("deleted")])
        merged_count = len([p for p in st.session_state.results if p.get("merged")])
        edge_dup_count = len([p for p in st.session_state.results if p.get("is_duplicate") and p.get("duplicate_type") == "edge_duplicate"])

        for idx, p in enumerate(st.session_state.results):
            # Categorize particle
            is_merged = p.get("merged", False)
            is_edge_dup = p.get("deleted") and p.get("duplicate_type") == "edge_duplicate"
            is_overlabeled = p.get("deleted") and p.get("duplicate_type") == "location_duplicate"
            is_deleted_only = p.get("deleted") and not is_overlabeled and not is_edge_dup
            is_active = not p.get("deleted")

            # Check if any filters are active
            any_filters_active = show_edge_dups or show_overlabeled or show_merged or show_deleted

            # Determine if particle should be included
            if any_filters_active:
                # Show only checked categories
                include = False
                if is_edge_dup and show_edge_dups:
                    include = True
                elif is_overlabeled and show_overlabeled:
                    include = True
                elif is_merged and show_merged:
                    include = True
                elif show_deleted and p.get("deleted"):
                    # Show ALL deleted particles
                    include = True
            else:
                # Default: show all active particles (including merged)
                include = is_active

            if not include:
                continue

            # Apply class/size filters to ALL particles
            if p.get("class") not in filter_class or p.get("size_bin") not in filter_bins:
                continue

            if show_seams_only and not p.get("at_seam"):
                continue
            if show_merged_only and not p.get("merged"):
                continue

            all_particles.append((idx, p))

    # Apply sorting
    if sort_by == "Confidence ↓":
        all_particles.sort(key=lambda x: x[1].get('confidence', 0), reverse=True)
    elif sort_by == "Confidence ↑":
        all_particles.sort(key=lambda x: x[1].get('confidence', 0), reverse=False)
    elif sort_by == "Size ↓":
        all_particles.sort(key=lambda x: x[1].get('diameter_um', 0), reverse=True)
    elif sort_by == "Size ↑":
        all_particles.sort(key=lambda x: x[1].get('diameter_um', 0), reverse=False)

    # If showing duplicates or over-labeled only, pair them with their matches/kept versions
    if show_edge_dups or show_overlabeled:
        paired_particles = []
        seen_matches = set()

        for idx, p in all_particles:
            is_edge_dup = p.get("deleted") and p.get("duplicate_type") == "edge_duplicate"
            is_overlabeled = p.get("deleted") and p.get("duplicate_type") == "location_duplicate"

            if show_edge_dups and is_edge_dup:
                # Skip if already paired
                if idx in seen_matches:
                    continue

                matched_idx = p.get("duplicate_match")
                if matched_idx is not None and matched_idx < len(st.session_state.results):
                    other_p = st.session_state.results[matched_idx]
                    paired_particles.append(((idx, p), (matched_idx, other_p)))
                    seen_matches.add(idx)
                    seen_matches.add(matched_idx)

            elif show_overlabeled and is_overlabeled:
                if idx in seen_matches:
                    continue

                # For over-labeling, find the kept particle at same location
                for kept_idx, kept_p in enumerate(st.session_state.results):
                    if kept_p.get("deleted"):
                        continue
                    if kept_idx in seen_matches:
                        continue

                    # Check if same location (within 20px)
                    dist = ((p['x'] - kept_p['x'])**2 + (p['y'] - kept_p['y'])**2)**0.5
                    if dist < 20:
                        paired_particles.append(((kept_idx, kept_p), (idx, p)))
                        seen_matches.add(kept_idx)
                        seen_matches.add(idx)
                        break

        # Info
        if show_edge_dups and len([p for p in paired_particles if p[0][1].get("duplicate_type") == "edge_duplicate"]) > 0:
            st.info(f"✅ Showing edge duplicate pairs")
        if show_overlabeled and len([p for p in paired_particles if p[0][1].get("duplicate_type") == "location_duplicate"]) > 0:
            st.info(f"✅ Showing over-labeled pairs")

        display_particles = paired_particles
    else:
        # Regular display (not paired)
        display_particles = all_particles

    # Determine if showing pairs BEFORE using in pagination
    is_showing_pairs = (show_edge_dups or show_overlabeled) and len(display_particles) > 0 and isinstance(display_particles[0], tuple) and isinstance(display_particles[0][0], tuple)

    if all_particles:
        # Show total count at top
        st.metric("Total Particles Displayed", len(all_particles) if not is_showing_pairs else len(display_particles) * 2)
        st.divider()

        # Select All button
        col_sel1, col_sel2 = st.columns([1, 1])
        with col_sel1:
            if st.button(f"✅ Select All ({len(all_particles)} displayed)", use_container_width=True, key="select_all_gallery"):
                # Set all checkbox states to True
                for idx, p in all_particles:
                    st.session_state[f"chk_{idx}"] = True
                st.success(f"✅ Selected {len(all_particles)} particles")
                st.rerun()
        with col_sel2:
            if st.button("⬜ Clear Page Selection", use_container_width=True, key="clear_selection_gallery"):
                st.success("✅ Use individual checkboxes to deselect")

        # Show count of selected on this page
        page_selected = sum(1 for idx, p in all_particles
                           if st.session_state.get(f"chk_{idx}", False))
        total_selected = sum(1 for idx in range(len(st.session_state.results))
                            if st.session_state.get(f"chk_{idx}", False))
        st.caption(f"On this page: {page_selected} selected | Total: {total_selected} selected")

        # DELETE SELECTED IN THIS VIEW button
        if page_selected > 0:
            if st.button(f"🗑️ Delete {page_selected} Selected (on this view)", use_container_width=True, key="delete_selected_view"):
                push_undo()
                count = 0
                # Delete only checked particles that are currently displayed
                for idx, p in all_particles:
                    if st.session_state.get(f"chk_{idx}", False):
                        st.session_state.results[idx]["deleted"] = True
                        count += 1
                st.success(f"✅ Deleted {count} selected particles from this view!")
                st.rerun()



        # Pagination - calculate based on what we're displaying
        if is_showing_pairs:
            # For pairs: pairs_per_page pairs = items_per_page items (2 particles per pair)
            pairs_per_page = max(1, items_per_page // 2)
            total_pairs = len(display_particles)
            total_pages = max(1, (total_pairs + pairs_per_page - 1) // pairs_per_page)

            if show_overlabeled:
                st.write(f"Showing {total_pairs} over-labeled pairs ({total_pairs * 2} items)")
            else:
                st.write(f"Showing {total_pairs} edge duplicate pairs ({total_pairs * 2} items)")
        else:
            # For regular: items_per_page particles per page
            total_pages = max(1, (len(display_particles) + items_per_page - 1) // items_per_page)

        # Show slider if multiple pages
        if total_pages > 1:
            page = st.slider("Page:", 1, total_pages, 1) - 1
        else:
            page = 0

        # Calculate slice for this page
        if is_showing_pairs:
            start = page * pairs_per_page
            end = start + pairs_per_page
        else:
            start = page * items_per_page
            end = start + items_per_page

        page_particles = []  # Initialize for full image viewer

        # RESET WIDGET COUNTER FOR NEW RENDER - ensures all keys are unique
        st.session_state.widget_counter = 0

        if is_showing_pairs:
            # Paired display: 2 columns per pair
            page_items = display_particles[start:end]

            for pair_idx, pair in enumerate(page_items):
                (idx1, p1), (idx2, p2) = pair
                page_particles.append((idx1, p1))
                page_particles.append((idx2, p2))

                # Show duplicate type indicator
                is_edge_dup = p1.get("duplicate_type") == "edge_duplicate"
                if is_edge_dup:
                    st.markdown("### 🔗 **EDGE DUPLICATE** (Particles at tile boundary)")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Particle 1** (Score: {p1.get('duplicate_score', 0)})")
                        display_particle_crop(idx1, p1)
                    with col2:
                        st.markdown(f"**Particle 2** (Score: {p2.get('duplicate_score', 0)})")
                        display_particle_crop(idx2, p2)
                else:
                    # Over-labeled: p1 is kept, p2 is deleted
                    st.markdown("### 🏷️ **TWICE LABELED** (Same spot, different class)")
                    col_kept, col_del = st.columns(2)
                    with col_kept:
                        st.markdown("**✅ KEPT**")
                        display_particle_crop(idx1, p1)
                    with col_del:
                        st.markdown("**❌ DELETED**")
                        display_particle_crop(idx2, p2)

                st.divider()
        else:
            # Regular gallery: 6 columns
            page_particles = display_particles[start:end]
            cols = st.columns(6)

            for i, (pidx, p) in enumerate(page_particles):
                with cols[i % 6]:
                    display_particle_crop(pidx, p)

        # Full image viewer - SKIP when showing pairs (they're already displayed inline)
        if not is_showing_pairs:
            # Find which particle has viewer open
            open_viewer = None
            for pidx in range(len(st.session_state.results)):
                if st.session_state.get(f"show_full_{pidx}", False):
                    open_viewer = pidx
                    break

            # Show close button if viewer is open
            if open_viewer is not None:
                if st.button("❌ Close Viewer", use_container_width=True, key="close_viewer_top"):
                    st.session_state[f"show_full_{open_viewer}"] = False
                    st.rerun()

            # Top action buttons
            col_top1, col_top2 = st.columns(2)
            with col_top1:
                if st.button("↶ Undo", use_container_width=True, key="undo_top"):
                    if st.session_state.undo_stack:
                        st.session_state.results = st.session_state.undo_stack.pop()
                        st.success("✅ Undo successful!")
                        st.rerun()
                    else:
                        st.warning("⚠️ Nothing to undo")

            with col_top2:
                if st.button("♻️ Restore All Deleted", use_container_width=True, key="restore_all_top"):
                    push_undo()
                    count = 0
                    for idx, p in enumerate(st.session_state.results):
                        if p.get("deleted"):
                            st.session_state.results[idx]["deleted"] = False
                            count += 1
                    st.success(f"✅ Restored {count} particles!")
                    st.rerun()


            # Show only one viewer at a time
            if open_viewer is not None:
                p = st.session_state.results[open_viewer]
                pidx = open_viewer
                try:
                    filename = p.get("tile_filename")
                    if not filename:
                        st.error("❌ Tile filename missing")
                    elif filename not in st.session_state.tile_files:
                        st.warning(f"❌ {filename} not found in loaded tiles")
                    else:
                        with st.expander(f"📸 {filename} - Particle #{pidx}", expanded=True):
                            st.markdown("---")
                            try:
                                # Check if this is a merged particle - show both pieces with boxes
                                if p.get("merged") and p.get("matched_stitch") is not None:
                                    matched_idx = p.get("matched_stitch")
                                    if matched_idx < len(st.session_state.results):
                                        p2 = st.session_state.results[matched_idx]
                                        stitch_edge = p.get("stitch_edge")

                                        st.subheader(f"🔀 Stitched Particle - {stitch_edge.upper()} Edge")

                                        # Get stitched preview
                                        stitched_img, merged_diameter_um = create_stitched_preview(st.session_state.tile_files, p, p2, stitch_edge)

                                        if stitched_img is not None:
                                            # Show stitched image
                                            fig = go.Figure()
                                            fig.add_trace(go.Image(z=stitched_img, name="Stitched"))
                                            fig.update_layout(height=700, showlegend=False, margin=dict(b=0, l=0, r=0, t=30))
                                            fig.update_xaxes(scaleanchor="y", scaleratio=1)
                                            fig.update_yaxes(scaleanchor="x", scaleratio=1)
                                            st.plotly_chart(fig, use_container_width=True)

                                            st.info(f"""
                                            **Stitched Particle Details:**
                                            - Stitch Edge: {stitch_edge.upper()}
                                            - Piece 1 Size: {p.get('diameter_um', 0):.1f}µm
                                            - Piece 2 Size: {p2.get('diameter_um', 0):.1f}µm
                                            - **Merged Size: {merged_diameter_um:.1f}µm**
                                            
                                            **Visual Guide:**
                                            - 🟠 Orange Box: Piece 1
                                            - 🟡 Yellow Box: Piece 2
                                            - 🔴 Red Line: Seam between tiles
                                            """)

                                            # Reject button
                                            col_rej, col_del, col_cls = st.columns(3)

                                            with col_rej:
                                                if st.button(f"❌ Reject Match", key=f"reject_viewer_{pidx}", use_container_width=True):
                                                    push_undo()
                                                    matched_idx = st.session_state.results[pidx].get("matched_stitch")
                                                    # Unmerge both particles in the pair
                                                    st.session_state.results[pidx]["merged"] = False
                                                    st.session_state.results[pidx]["matched_stitch"] = None
                                                    # Also unmerge the matched particle
                                                    if matched_idx is not None and matched_idx < len(st.session_state.results):
                                                        st.session_state.results[matched_idx]["merged"] = False
                                                        st.session_state.results[matched_idx]["matched_stitch"] = None
                                                    st.success("✅ Match rejected! Both particles unmarked.")
                                                    st.rerun()

                                            with col_del:
                                                if st.button(f"🗑️ Delete", key=f"del_viewer_{pidx}", use_container_width=True):
                                                    push_undo()
                                                    st.session_state.results[pidx]["deleted"] = True
                                                    st.success("Deleted!")
                                                    st.rerun()

                                            with col_cls:
                                                current_class = p.get("class", "Other")
                                                new_cls = st.selectbox(
                                                    "Class:",
                                                    ["Fiber", "Glass", "Metallic", "Other"],
                                                    index=["Fiber", "Glass", "Metallic", "Other"].index(current_class),
                                                    key=f"cls_viewer_{pidx}"
                                                )
                                                if new_cls != current_class:
                                                    push_undo()
                                                    st.session_state.results[pidx]["class"] = new_cls
                                                    st.rerun()

                                        st.markdown("---")

                                file_obj = st.session_state.tile_files[filename]
                                tile_img = Image.open(file_obj).convert('RGB')
                                tile_img = np.array(tile_img)

                                # Show single tile with particle bbox
                                st.subheader("📸 Original Tile")
                                fig = go.Figure()
                                fig.add_trace(go.Image(z=tile_img, name="Image"))
                                x, y, w, h = p.get("x", 0), p.get("y", 0), p.get("w", 0), p.get("h", 0)
                                if x and y and w and h:
                                    fig.add_shape(type="rect", x0=x, y0=y, x1=x+w, y1=y+h, line=dict(color="rgb(255, 100, 0)", width=2))
                                fig.update_layout(height=600, showlegend=False, margin=dict(b=0, l=0, r=0, t=30))
                                fig.update_xaxes(scaleanchor="y", scaleratio=1)
                                fig.update_yaxes(scaleanchor="x", scaleratio=1)
                                st.plotly_chart(fig, use_container_width=True)

                                st.markdown("---")
                                st.subheader("📏 Measure Particle")

                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("**Point 1 (Start):**")
                                    x1 = st.number_input("X₁:", min_value=0, value=int(x) if x else 0, key=f"x1_{pidx}")
                                    y1 = st.number_input("Y₁:", min_value=0, value=int(y) if y else 0, key=f"y1_{pidx}")

                                with col2:
                                    st.write("**Point 2 (End):**")
                                    x2 = st.number_input("X₂:", min_value=0, value=int(x+w) if (x and w) else 100, key=f"x2_{pidx}")
                                    y2 = st.number_input("Y₂:", min_value=0, value=int(y+h) if (y and h) else 100, key=f"y2_{pidx}")

                                # Calculate distance
                                dist_px = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                                dist_um = dist_px * CALIBRATION_UM_PER_PIXEL

                                st.markdown("---")
                                col_result1, col_result2, col_result3 = st.columns(3)
                                with col_result1:
                                    st.metric("📏 Distance (px)", f"{dist_px:.1f}")
                                with col_result2:
                                    st.metric("📏 Diameter (µm)", f"{dist_um:.2f}")
                                with col_result3:
                                    st.metric("Size Bin", get_size_bin(dist_um))

                                if st.button("✅ APPLY RESIZE", key=f"apply_{pidx}", use_container_width=True):
                                    push_undo()
                                    st.session_state.results[pidx]["diameter_um"] = dist_um
                                    st.session_state.results[pidx]["size_bin"] = get_size_bin(dist_um)
                                    st.session_state.results[pidx]["size_method"] = "manual_coords"
                                    st.session_state[f"show_full_{pidx}"] = False
                                    st.success(f"✅ Resized to {dist_um:.1f}µm!")
                                    st.rerun()
                            except Exception as e:
                                st.error(f"❌ {str(e)[:60]}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)[:60]}")






st.divider()
st.caption("💾 Session saved")