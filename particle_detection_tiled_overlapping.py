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
# Initialize Session State
# ============================================================
if 'results' not in st.session_state:
    st.session_state.results = []
if 'tile_metadata' not in st.session_state:
    st.session_state.tile_metadata = []
if 'tile_files' not in st.session_state:
    st.session_state.tile_files = {}
if 'selected_particles' not in st.session_state:
    st.session_state.selected_particles = set()
if 'undo_stack' not in st.session_state:
    st.session_state.undo_stack = []
if 'widget_counter' not in st.session_state:
    st.session_state.widget_counter = 0
if 'stitch_cache' not in st.session_state:
    st.session_state.stitch_cache = {}
if 'pipeline_stats' not in st.session_state:
    st.session_state.pipeline_stats = {}

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
    Simple tile-focused dedup:
    1. IOU dedup - remove overlapping particles in same tile
    2. Neighbor scoring - score edge particles against neighbors
    3. Over-labeling - same spot, different class

    Returns particles with:
    - deleted=True for overlaps and over-labeling
    - is_duplicate=True with score for edge particles matching neighbors
    """
    particles = [p.copy() for p in raw_particles]

    # Initialize deleted field for all particles
    for p in particles:
        if 'deleted' not in p:
            p['deleted'] = False

    # If no tile metadata, just do over-labeling
    if not tile_metadata:
        st.write("**No manifest - over-labeling detection only**")
        overlabel_removed = 0

        for i, p1 in enumerate(particles):
            if p1.get('deleted'):
                continue

            for j, p2 in enumerate(particles[i+1:], start=i+1):
                if p2.get('deleted'):
                    continue

                # Different class
                if p1.get('class') == p2.get('class'):
                    continue

                # Same spot (within 20px)
                cx1 = p1.get('x', 0) + p1.get('w', 0)/2
                cy1 = p1.get('y', 0) + p1.get('h', 0)/2
                cx2 = p2.get('x', 0) + p2.get('w', 0)/2
                cy2 = p2.get('y', 0) + p2.get('h', 0)/2

                dist = ((cx1 - cx2)**2 + (cy1 - cy2)**2)**0.5

                if dist < 20:
                    if p1['confidence'] > p2['confidence']:
                        particles[j]['deleted'] = True
                        particles[j]['duplicate_type'] = 'location_duplicate'
                        overlabel_removed += 1
                    else:
                        particles[i]['deleted'] = True
                        particles[i]['duplicate_type'] = 'location_duplicate'
                        overlabel_removed += 1
                        break

        st.write(f"  ✅ Removed {overlabel_removed} over-labeled particles")
        return particles, {"iou_removed": 0, "edge_duplicates_found": 0, "overlabel_removed": overlabel_removed}

    # Build neighbor map
    neighbors_map = {}
    for tile in tile_metadata:
        fname = tile.get('filename')
        neighbors_map[fname] = tile.get('neighbors', {})

    # Step 1: IOU DEDUP (within same tile)
    st.write("**Step 1: IOU Dedup (overlapping particles)**")

    iou_removed = 0

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

            # Only compare if same tile
            if p1.get('tile_filename') != p2.get('tile_filename'):
                continue

            # Size & class must match
            if p1.get('class') != p2.get('class'):
                continue
            if abs(p1.get('diameter_um', 0) - p2.get('diameter_um', 0)) / max(p1.get('diameter_um', 1), p2.get('diameter_um', 1)) > 0.2:
                continue

            # Check IOU
            box1 = (p1['x'], p1['y'], p1['x'] + p1['w'], p1['y'] + p1['h'])
            box2 = (p2['x'], p2['y'], p2['x'] + p2['w'], p2['y'] + p2['h'])

            if iou_2d(box1, box2) > 0.3:
                # Delete lower confidence
                if p1['confidence'] > p2['confidence']:
                    particles[j]['deleted'] = True
                    particles[j]['duplicate_type'] = 'overlap_duplicate'
                    particles[j]['duplicate_reason'] = f"IOU overlap with higher confidence"
                    iou_removed += 1
                else:
                    particles[i]['deleted'] = True
                    particles[i]['duplicate_type'] = 'overlap_duplicate'
                    particles[i]['duplicate_reason'] = f"IOU overlap with higher confidence"
                    iou_removed += 1
                    break

    st.write(f"  ✅ Removed {iou_removed} overlapping particles")

    # Step 2: NEIGHBOR EDGE SCORING
    st.write("**Step 2: Edge particle scoring vs neighbors**")

    edge_dup_count = 0

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

        tile_w = particle.get('tile_width', 1024)
        tile_h = particle.get('tile_height', 1024)
        edge_margin = 50

        at_edges = []
        if cx < edge_margin: at_edges.append('left')
        if cx > tile_w - edge_margin: at_edges.append('right')
        if cy < edge_margin: at_edges.append('top')
        if cy > tile_h - edge_margin: at_edges.append('bottom')

        if not at_edges:
            continue

        # Score against neighbors
        for edge in at_edges:
            neighbor_file = neighbors_map[tile_file].get(edge)
            if not neighbor_file:
                continue

            best_score = 0
            best_match = None

            for nidx, neighbor in enumerate(particles):
                if neighbor.get('deleted') or neighbor.get('tile_filename') != neighbor_file:
                    continue

                # Class match
                class_score = 1.0 if particle.get('class') == neighbor.get('class') else 0.3

                # Size match
                d1 = particle.get('diameter_um', 50)
                d2 = neighbor.get('diameter_um', 50)
                size_score = max(0, 1.0 - abs(d1 - d2) / max(d1, d2))

                # Position match (perpendicular to boundary)
                if edge in ['left', 'right']:
                    y_diff = abs(cy - (neighbor['y'] + neighbor['h']/2))
                    y_max = max(h, neighbor['h']) * 2
                    pos_score = max(0, 1.0 - (y_diff / y_max))
                else:
                    x_diff = abs(cx - (neighbor['x'] + neighbor['w']/2))
                    x_max = max(w, neighbor['w']) * 2
                    pos_score = max(0, 1.0 - (x_diff / x_max))

                score = (class_score * 0.3) + (size_score * 0.3) + (pos_score * 0.4)

                if score > best_score:
                    best_score = score
                    best_match = nidx

            # Mark if good match
            if best_score >= 0.70 and best_match is not None:
                particles[pidx]['is_duplicate'] = True
                particles[pidx]['duplicate_type'] = 'edge_duplicate'
                particles[pidx]['duplicate_match'] = best_match
                particles[pidx]['duplicate_score'] = round(best_score, 3)
                particles[pidx]['matched_edge'] = edge
                particles[pidx]['at_seam'] = True
                edge_dup_count += 1
                break

    st.write(f"  ✅ Found {edge_dup_count} edge duplicate candidates")

    # Step 3: OVER-LABELING DETECTION (same spot, different class)
    st.write("**Step 3: Over-labeling detection (same spot, different class)**")

    overlabel_removed = 0

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

            if dist < 20:
                # Delete lower confidence
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

    return particles, {"iou_removed": iou_removed, "edge_duplicates_found": edge_dup_count, "overlabel_removed": overlabel_removed}


    """
    Integrated edge dedup - finds tile boundary duplicates.
    Marks particles with is_duplicate=True (doesn't delete them).
    """
    if not manifest or not results:
        return results

    from collections import defaultdict

    # Group by tile
    by_tile = defaultdict(list)
    for pidx, p in enumerate(results):
        tile = p.get('tile_filename', 'unknown')
        by_tile[tile].append((pidx, p))

    # Find neighbors in manifest
    neighbors_map = {}
    for tile_info in manifest.get('tiles', []):
        fname = tile_info.get('filename')
        neighbors_map[fname] = tile_info.get('neighbors', {})

    # Check each particle
    for pidx, particle in enumerate(results):
        if particle.get('deleted') or particle.get('is_duplicate'):
            continue

        tile_file = particle.get('tile_filename')
        tw = particle.get('tile_width', 1024)
        th = particle.get('tile_height', 1024)

        # Check if at edge
        x, y = particle.get('x', 0), particle.get('y', 0)
        w, h = particle.get('w', 0), particle.get('h', 0)
        cx, cy = x + w/2, y + h/2

        at_edges = []
        if cx < edge_margin: at_edges.append('left')
        if cx > tw - edge_margin: at_edges.append('right')
        if cy < edge_margin: at_edges.append('top')
        if cy > th - edge_margin: at_edges.append('bottom')

        if not at_edges:
            continue

        # Look for match in neighbor
        for edge in at_edges:
            neighbor_tile = neighbors_map.get(tile_file, {}).get(edge)
            if not neighbor_tile:
                continue

            neighbor_particles = by_tile.get(neighbor_tile, [])

            # Score each neighbor particle
            best_score = 0
            best_idx = None

            for nidx, neighbor in neighbor_particles:
                if neighbor.get('deleted') or neighbor.get('is_duplicate'):
                    continue

                # Check if neighbor also at edge
                nx, ny = neighbor.get('x', 0), neighbor.get('y', 0)
                nw, nh = neighbor.get('w', 0), neighbor.get('h', 0)
                ncx, ncy = nx + nw/2, ny + nh/2

                neighbor_edges = []
                if ncx < edge_margin: neighbor_edges.append('left')
                if ncx > tw - edge_margin: neighbor_edges.append('right')
                if ncy < edge_margin: neighbor_edges.append('top')
                if ncy > th - edge_margin: neighbor_edges.append('bottom')

                # Opposite edge
                opposite = {'left': 'right', 'right': 'left', 'top': 'bottom', 'bottom': 'top'}
                if opposite[edge] not in neighbor_edges:
                    continue

                # Score it
                class_score = 1.0 if particle.get('class') == neighbor.get('class') else 0.3

                d1 = particle.get('diameter_um', 50)
                d2 = neighbor.get('diameter_um', 50)
                size_score = max(0, 1.0 - abs(d1 - d2) / max(d1, d2))

                # Position alignment
                if edge in ['left', 'right']:
                    # Compare Y
                    y_diff = abs(cy - ncy)
                    y_max = max(h, nh) * 2
                    pos_score = max(0, 1.0 - (y_diff / y_max))
                else:
                    # Compare X
                    x_diff = abs(cx - ncx)
                    x_max = max(w, nw) * 2
                    pos_score = max(0, 1.0 - (x_diff / x_max))

                final_score = (class_score * 0.3) + (size_score * 0.3) + (pos_score * 0.4)

                if final_score > best_score:
                    best_score = final_score
                    best_idx = nidx

            # Mark if good match
            if best_score >= score_threshold and best_idx is not None:
                results[pidx]['is_duplicate'] = True
                results[pidx]['duplicate_type'] = 'edge_duplicate'
                results[pidx]['duplicate_match'] = best_idx
                results[pidx]['duplicate_score'] = round(best_score, 3)
                results[pidx]['matched_edge'] = edge
                results[pidx]['at_seam'] = True
                break

    return results


    for label, lo, hi in SIZE_BINS:
        if lo <= diameter_um < hi:
            return label
    return "K"

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

def push_undo():
    st.session_state.undo_stack.append(deepcopy(st.session_state.results))

def display_particle_crop(pidx, p):
    """Display a single particle as a gallery thumbnail with ALL interactive features"""
    try:
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
            key = f"sel_{widget_id}"
            is_selected = key in st.session_state.selected_particles
            if st.checkbox("Select", value=is_selected, key=key):
                st.session_state.selected_particles.add(key)
            else:
                st.session_state.selected_particles.discard(key)

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

                    # Summary
                    st.divider()
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Raw Detections", len(raw_particles))
                    col2.metric("Overlaps Removed", dedup_stats['iou_removed'])
                    col3.metric("Over-labeled Removed", dedup_stats['overlabel_removed'])
                    col4.metric("Edge Duplicates Found", dedup_stats['edge_duplicates_found'])

                    st.success("✅ Processing complete!")
                except Exception as e:
                    st.error(f"Pipeline error: {e}")
                    import traceback
                    st.error(traceback.format_exc())
                    st.session_state.results = raw_particles

                st.session_state.undo_stack = []
                st.session_state.selected_particles = set()
                st.success(f"✅ Done! Check galleries below.")
    if st.session_state.results:
        total = len([p for p in st.session_state.results if not p.get("deleted")])
        st.success(f"✅ {total} particles")
        st.write(f"**Selected:** {len(st.session_state.selected_particles)}")

    st.divider()

    if st.button("📥 Export CSV"):
        if st.session_state.results:
            rows = []
            for p in st.session_state.results:
                if not p.get("deleted"):
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

    totals_row = {"Material": "TOTAL"}
    for b, _, _ in SIZE_BINS:
        total = sum(data[cls][b] for cls in ["Fiber", "Glass", "Metallic", "Other"])
        totals_row[b] = total
    rows.append(totals_row)
    
    for cls in ["Fiber", "Glass", "Metallic", "Other"]:
        row = {"Material": cls}
        for b, _, _ in SIZE_BINS:
            c = data[cls][b]
            row[b] = c
        rows.append(row)

    # Add totals row


    df = pd.DataFrame(rows)
    st.dataframe(df, width='stretch', height=200)

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # GALLERY
    # ─────────────────────────────────────────────────────────────────────────

    st.subheader("🖼️ Particle Gallery")

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
        show_active = st.checkbox("Active", value=True)
    with col_s2:
        show_edge_dups = st.checkbox("🔗 Edge Dups", value=True)
    with col_s3:
        show_overlabeled = st.checkbox("🏷️ Over-Labeled", value=False)
    with col_s4:
        show_deleted = st.checkbox("❌ Deleted", value=False)

    # Show filter info
    st.caption(f"💡 **Edge Duplicates** = particles at tile seams | **Over-Labeled** = same spot, different class")

    # Filter particles
    all_particles = []
    if st.session_state.results is not None:
        # Debug: count particles by status
        total = len(st.session_state.results)
        deleted_count = len([p for p in st.session_state.results if p.get("deleted")])
        merged_count = len([p for p in st.session_state.results if p.get("merged")])
        edge_dup_count = len([p for p in st.session_state.results if p.get("is_duplicate") and p.get("duplicate_type") == "edge_duplicate"])

        for idx, p in enumerate(st.session_state.results):
            # Filter by duplicate type based on checkboxes
            is_active = not p.get("deleted") and not (p.get("is_duplicate") and p.get("duplicate_type") == "edge_duplicate")
            is_edge_dup = p.get("is_duplicate") and p.get("duplicate_type") == "edge_duplicate"
            is_overlabeled = p.get("deleted") and p.get("duplicate_type") == "location_duplicate"
            is_any_deleted = p.get("deleted")

            # Check which category this particle falls into
            include = False
            if is_active and show_active:
                include = True
            elif is_edge_dup and show_edge_dups:
                include = True
            elif is_overlabeled and show_overlabeled:
                include = True
            elif is_any_deleted and show_deleted and not is_overlabeled:
                # Show other deleted particles (not over-labeled)
                include = True

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
            is_edge_dup = p.get("is_duplicate") and p.get("duplicate_type") == "edge_duplicate"
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
            for pidx, p in [(idx, p) for idx, p in page_particles]:
                if st.session_state.get(f"show_full_{pidx}", False):
                    try:
                        filename = p.get("tile_filename")
                        if not filename:
                            st.error("❌ Tile filename missing")
                            continue

                        if filename not in st.session_state.tile_files:
                            st.warning(f"❌ {filename} not found")
                        else:
                            with st.expander(f"📸 {filename}", expanded=True):
                                # Close button
                                if st.button("❌ Close", key=f"close_view_{pidx}", use_container_width=True):
                                    st.session_state[f"show_full_{pidx}"] = False
                                    st.rerun()

                                st.markdown("---")
                                try:
                                    file_obj = st.session_state.tile_files[filename]
                                    tile_img = Image.open(file_obj).convert('RGB')
                                    tile_img = np.array(tile_img)

                                    # Show image with particle bbox
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


# ─────────────────────────────────────────────────────────────────────────────
# MASS EDIT / BULK OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.selected_particles:
    st.divider()
    st.subheader("⚙️ Bulk Operations")

    col1, col2, col3 = st.columns(3)
    with col1:
        action = st.selectbox("Action:", ["Delete", "Change Class", "Clear All"])
    with col2:
        if action == "Change Class":
            new_cls = st.selectbox("To Class:", ["Fiber", "Glass", "Metallic", "Other"])

    if st.button(f"Apply to {len(st.session_state.selected_particles)} particles", use_container_width=True):
        try:
            for pidx in list(st.session_state.selected_particles):
                if action == "Delete":
                    st.session_state.results[pidx]["deleted"] = True
                elif action == "Change Class":
                    st.session_state.results[pidx]["class"] = new_cls
                elif action == "Clear All":
                    st.session_state.selected_particles.clear()

            st.success(f"✅ {action} applied!")
            st.session_state.selected_particles.clear()
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error: {str(e)[:100]}")

st.divider()
st.caption("💾 Session saved")