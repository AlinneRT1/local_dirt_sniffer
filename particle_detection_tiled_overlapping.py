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

        # Mark deleted/duplicate particles
        if p.get("deleted"):
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
        if p.get("at_seam") and not p.get("merged"):
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

    # DEBUG SECTION FOR EDGE DETECTION
    st.subheader("🔍 Debug: Edge Detection")

    if st.button("Check Edge Particles"):
        if not st.session_state.results:
            st.warning("⚠️ Run detection first!")
        elif not st.session_state.tile_metadata:
            st.warning("⚠️ Load tiles first!")
        else:
            particles = st.session_state.results
            tile_metadata = st.session_state.tile_metadata

            # Build tile info dict using index as tile_id
            tiles_dict = {}
            for tile_idx, tm in enumerate(tile_metadata):
                tiles_dict[tile_idx] = {
                    'width': tm.get('width', 0),
                    'height': tm.get('height', 0)
                }

            # Count particles at edges
            total_at_edges = 0
            edge_summary = {}

            st.write("**Analysis:**")
            st.write(f"  Total particles: {len(particles)}")

            # Check each particle
            for p in particles:
                tile_id = p.get('tile_id', 0)
                if tile_id not in tiles_dict:
                    continue

                tile_w = tiles_dict[tile_id].get('width', 0)
                tile_h = tiles_dict[tile_id].get('height', 0)

                x, y, w, h = p.get('x', 0), p.get('y', 0), p.get('w', 0), p.get('h', 0)
                margin = 50

                # Check if at any edge
                at_edge = False
                edge_type = []

                if x < margin:
                    at_edge = True
                    edge_type.append("left")
                if (x + w) > (tile_w - margin):
                    at_edge = True
                    edge_type.append("right")
                if y < margin:
                    at_edge = True
                    edge_type.append("top")
                if (y + h) > (tile_h - margin):
                    at_edge = True
                    edge_type.append("bottom")

                if at_edge:
                    total_at_edges += 1
                    edges = ",".join(edge_type)
                    key = f"Tile {tile_id} ({edges})"
                    edge_summary[key] = edge_summary.get(key, 0) + 1

            # Display results
            st.write(f"  **At edges (50px margin):** {total_at_edges}")
            if len(particles) > 0:
                pct = 100 * total_at_edges / len(particles)
                st.write(f"  **Percentage:** {pct:.1f}%")

            if total_at_edges == 0:
                st.warning("⚠️ No particles at edges!")
                with st.expander("Why?"):
                    st.write("""
                    **Possible causes:**
                    - Tiles don't overlap
                    - Particles all in centers
                    - Edge margin (50px) too small
                    
                    **To fix:**
                    1. Check if your tiles actually overlap
                    2. If yes, increase margin in intelligent_particle_matcher.py line 47:
                       `margin = 50` → `margin = 100 or 150`
                    """)
            else:
                st.success(f"✅ Found {total_at_edges} particles at edges!")
                with st.expander("Breakdown by tile"):
                    for key, count in sorted(edge_summary.items()):
                        st.write(f"  {key}: {count}")

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

                # Step 2-4: PROPER DEDUP & STITCH PIPELINE
                st.divider()
                st.write("## 🔄 Processing Pipeline")

                try:
                    from dedup_stitch_pipeline import run_full_dedup_and_stitch_pipeline

                    final_particles, pipeline_stats = run_full_dedup_and_stitch_pipeline(
                        raw_particles,
                        st.session_state.tile_metadata,
                        st.session_state.tile_files
                    )

                    st.session_state.results = final_particles
                    st.session_state.pipeline_stats = pipeline_stats

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
        show_duplicates_only = st.checkbox("Show duplicates only")

    # Over-labeled filter row
    col_ol1, col_ol2 = st.columns([1, 5])
    with col_ol1:
        show_overlabel_only = st.checkbox("Over-labeled only")

    # Pagination row
    col_p1, col_p2, col_p3, col_p4, col_p5, col_p6 = st.columns(6)
    with col_p6:
        items_per_page = st.selectbox("Per page:", [12, 18, 24, 36], index=0)

    # Filter particles
    all_particles = []
    if st.session_state.results is not None:
        # Debug: count particles by status
        total = len(st.session_state.results)
        deleted_count = len([p for p in st.session_state.results if p.get("deleted")])
        merged_count = len([p for p in st.session_state.results if p.get("merged")])

        # Show debug info in expander
        with st.expander("🔍 Debug - Particle Status Count"):
            st.write(f"Total particles: {total}")
            st.write(f"Deleted particles: {deleted_count}")
            st.write(f"Merged particles: {merged_count}")
            st.write(f"Active particles: {total - deleted_count}")

        for idx, p in enumerate(st.session_state.results):
            # Filter by deleted status
            if show_overlabel_only:
                # Show ONLY over-labeled particles (deleted AND location_duplicate type)
                if not p.get("deleted") or p.get("duplicate_type") != "location_duplicate":
                    continue
            elif show_duplicates_only:
                # Show ONLY deleted particles - NO OTHER FILTERS
                if not p.get("deleted"):
                    continue
                # Don't apply class/size filters for duplicates - show ALL deleted particles
            else:
                # Show only ACTIVE particles (default)
                if p.get("deleted"):
                    continue

                # Apply class/size filters ONLY for active particles
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
    if show_duplicates_only or show_overlabel_only:
        paired_particles = []
        seen_matches = set()
        unmatched_count = 0

        for idx, p in all_particles:
            if not p.get("deleted"):
                continue

            # For duplicates: match via matched_with field
            # For over-labeling: look for particles at same location
            if show_duplicates_only:
                matched_idx = p.get("matched_with")
                if matched_idx is not None and matched_idx not in seen_matches:
                    kept_p = None
                    if matched_idx < len(st.session_state.results):
                        kept_p = st.session_state.results[matched_idx]

                    if kept_p:
                        paired_particles.append(((matched_idx, kept_p), (idx, p)))
                        seen_matches.add(matched_idx)
                else:
                    unmatched_count += 1

            elif show_overlabel_only:
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
                        break
                else:
                    unmatched_count += 1

        # Debug info
        if show_duplicates_only:
            st.info(f"✅ Found {len(paired_particles)} duplicate pairs | ❌ {unmatched_count} deleted particles without match link")
        else:
            st.info(f"✅ Found {len(paired_particles)} over-labeled pairs | ❌ {unmatched_count} deleted particles without match link")

        display_particles = paired_particles
    else:
        # Regular display (not paired)
        display_particles = all_particles

    # Determine if showing pairs BEFORE using in pagination
    is_showing_pairs = (show_duplicates_only or show_overlabel_only) and len(display_particles) > 0 and isinstance(display_particles[0], tuple) and isinstance(display_particles[0][0], tuple)

    if all_particles:
        # Pagination - calculate based on what we're displaying
        if is_showing_pairs:
            # For pairs: pairs_per_page pairs = items_per_page items (2 particles per pair)
            pairs_per_page = max(1, items_per_page // 2)
            total_pairs = len(display_particles)
            total_pages = max(1, (total_pairs + pairs_per_page - 1) // pairs_per_page)

            pair_type = "over-labeled" if show_overlabel_only else "duplicate"
            st.write(f"Showing {total_pairs} {pair_type} pairs ({total_pairs * 2} items)")
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
                (kept_idx, kept_p), (del_idx, del_p) = pair
                page_particles.append((kept_idx, kept_p))  # Add to page_particles for full viewer
                page_particles.append((del_idx, del_p))

                # Show duplicate type indicator
                dup_type = del_p.get("duplicate_type", "unknown")
                if dup_type == "edge_duplicate":
                    st.markdown("### 🔪 **EDGE DUPLICATE** (Cut across tile boundary)")
                elif dup_type == "location_duplicate":
                    st.markdown("### 🏷️ **TWICE LABELED** (Same spot, different class)")
                else:
                    st.markdown("### ❌ **DUPLICATE**")

                col_kept, col_del = st.columns(2)

                # KEPT particle (left column)
                with col_kept:
                    st.markdown("**✅ KEPT**")
                    display_particle_crop(kept_idx, kept_p)

                # DELETED particle (right column)
                with col_del:
                    st.markdown("**❌ DELETED**")
                    display_particle_crop(del_idx, del_p)

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

                        # Check if merged - show stitched image with red boundary
                        if p.get("merged") and p.get("stitched_image") is not None:
                            with st.expander(f"🔗 STITCHED PARTICLE: {p.get('tile_filename', '?')}", expanded=True):
                                st.info("✅ Cut particle detected and merged")

                                try:
                                    # Display the pre-stitched image (has red boundary line AND blue bounding box) ✅
                                    stitched_img = p.get("stitched_image")
                                    if isinstance(stitched_img, np.ndarray):
                                        # Convert to Plotly for interactive zoom
                                        stitched_pil = Image.fromarray(stitched_img.astype(np.uint8))
                                        stitched_array = np.array(stitched_pil)

                                        # Create interactive Plotly figure
                                        fig_stitched = go.Figure()
                                        fig_stitched.add_trace(go.Image(z=stitched_array, name="Stitched Particle"))

                                        fig_stitched.update_layout(
                                            title="🔗 Stitched Particle (Click & Drag to Zoom, Double-click to Reset)",
                                            showlegend=False,
                                            hovermode="closest",
                                            margin=dict(b=0, l=0, r=0, t=50),
                                            height=700,
                                            xaxis=dict(showgrid=False),
                                            yaxis=dict(showgrid=False)
                                        )
                                        fig_stitched.update_xaxes(scaleanchor="y", scaleratio=1)
                                        fig_stitched.update_yaxes(scaleanchor="x", scaleratio=1, autorange="reversed")

                                        st.plotly_chart(fig_stitched)

                                        st.markdown(f"**🔴 Red line at x≈{p.get('boundary_line_pos', '?')}px = tile boundary | 🔵 Blue box = particle extent**")

                                        # SIZING BOX - Show measurements
                                        st.markdown("---")
                                        st.markdown("### 📐 Measurements & Pixel Details")

                                        col1, col2, col3, col4 = st.columns(4)
                                        with col1:
                                            st.metric("Original Diameter", f"{p.get('original_diameter_um', '?'):.1f}µm")
                                        with col2:
                                            st.metric("Stitched Diameter", f"{p.get('diameter_um', '?'):.1f}µm")
                                        with col3:
                                            st.metric("Size Change", f"{p.get('size_change_pct', 0):+.1f}%")
                                        with col4:
                                            st.metric("Match Score", f"{p.get('match_score', 0):.3f}")

                                        # More details
                                        st.markdown("---")
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.write(f"**Original Bbox:** {p.get('w', 0)}×{p.get('h', 0)}px")
                                        with col2:
                                            st.write(f"**Stitched Bbox:** {p.get('stitched_w', 0)}×{p.get('stitched_h', 0)}px")
                                        with col3:
                                            original_diag = np.sqrt(p.get('w', 0)**2 + p.get('h', 0)**2) if p.get('w', 0) > 0 else 0
                                            st.write(f"**Diagonal:** {original_diag:.0f}px → {p.get('stitched_diagonal_px', 0):.0f}px")

                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.write(f"**Overlap:** {p.get('overlap_pixels', 0)}px")
                                        with col2:
                                            st.write(f"**Direction:** {p.get('stitch_direction', '?')}")
                                        with col3:
                                            st.write(f"**Class:** {p.get('class', '?')} | {p.get('size_bin', '?')}")

                                        # LINE MEASUREMENT TOOL FOR STITCHED - SIMPLE CLICKABLE IMAGE
                                        st.markdown("---")
                                        st.subheader("📏 Resize Stitched Particle")
                                        st.write("**Just click on the image twice: first click = red dot, second click = green dot**")

                                        measure_key_s = f"measure_stitch_{pidx}"
                                        if measure_key_s not in st.session_state:
                                            st.session_state[measure_key_s] = []

                                        points_s = st.session_state[measure_key_s]

                                        # Create canvas with JavaScript click detection
                                        img_pil_s = Image.fromarray(stitched_img.astype(np.uint8))
                                        buf_s = io.BytesIO()
                                        img_pil_s.save(buf_s, format="PNG")
                                        img_base64_s = base64.b64encode(buf_s.getvalue()).decode()

                                        html_canvas_s = f"""
                                        <style>
                                        #measureCanvasStitch {{
                                            cursor: crosshair;
                                            border: 3px solid #ccc;
                                            display: block;
                                            max-width: 100%;
                                            height: auto;
                                        }}
                                        #coordsStitch {{
                                            font-family: monospace;
                                            margin-top: 10px;
                                            font-size: 14px;
                                            color: #333;
                                        }}
                                        </style>
                                        <canvas id="measureCanvasStitch" width="{stitched_img.shape[1]}" height="{stitched_img.shape[0]}"></canvas>
                                        <p id="coordsStitch">Click image to add points (need 2)</p>
                                        <button id="confirmBtnStitch" style="margin-top: 10px; padding: 10px 20px; font-size: 14px; cursor: pointer; background: #4CAF50; color: white; border: none; border-radius: 4px;">Confirm Points</button>
                                        <script>
                                        const canvasS = document.getElementById('measureCanvasStitch');
                                        const ctxS = canvasS.getContext('2d');
                                        const coordsElS = document.getElementById('coordsStitch');
                                        const confirmBtnS = document.getElementById('confirmBtnStitch');
                                        const imgS = new Image();
                                        
                                        let clickPointsS = [];
                                        
                                        imgS.onload = function() {{
                                            ctxS.drawImage(imgS, 0, 0);
                                            redrawS();
                                        }};
                                        imgS.src = 'data:image/png;base64,{img_base64_s}';
                                        
                                        function redrawS() {{
                                            ctxS.drawImage(imgS, 0, 0);
                                            
                                            // Draw existing points
                                            clickPointsS.forEach((p, i) => {{
                                                const color = i === 0 ? 'red' : 'green';
                                                ctxS.fillStyle = color;
                                                ctxS.beginPath();
                                                ctxS.arc(p.x, p.y, 12, 0, 2*Math.PI);
                                                ctxS.fill();
                                                ctxS.strokeStyle = 'white';
                                                ctxS.lineWidth = 2;
                                                ctxS.stroke();
                                            }});
                                            
                                            // Draw line if 2 points
                                            if (clickPointsS.length === 2) {{
                                                ctxS.strokeStyle = 'orange';
                                                ctxS.lineWidth = 4;
                                                ctxS.beginPath();
                                                ctxS.moveTo(clickPointsS[0].x, clickPointsS[0].y);
                                                ctxS.lineTo(clickPointsS[1].x, clickPointsS[1].y);
                                                ctxS.stroke();
                                            }}
                                        }}
                                        
                                        canvasS.addEventListener('click', function(e) {{
                                            if (clickPointsS.length >= 2) return;
                                            
                                            const rect = canvasS.getBoundingClientRect();
                                            const x = Math.round((e.clientX - rect.left) * canvasS.width / rect.width);
                                            const y = Math.round((e.clientY - rect.top) * canvasS.height / rect.height);
                                            
                                            clickPointsS.push({{x: x, y: y}});
                                            
                                            if (clickPointsS.length === 1) {{
                                                coordsElS.textContent = `Point 1: (${{x}}, ${{y}}) - click again for point 2`;
                                            }} else {{
                                                const dx = clickPointsS[1].x - clickPointsS[0].x;
                                                const dy = clickPointsS[1].y - clickPointsS[0].y;
                                                const dist = Math.sqrt(dx*dx + dy*dy);
                                                coordsElS.textContent = `Points ready! Distance: ${{dist.toFixed(1)}}px | Click "Confirm Points" button`;
                                            }}
                                            
                                            redrawS();
                                        }});
                                        
                                        confirmBtnS.addEventListener('click', function() {{
                                            if (clickPointsS.length === 2) {{
                                                window.confirmedPointsStitch_{pidx} = clickPointsS;
                                                coordsElS.textContent = `✅ Points confirmed: (${{clickPointsS[0].x}}, ${{clickPointsS[0].y}}) → (${{clickPointsS[1].x}}, ${{clickPointsS[1].y}})`;
                                                confirmBtnS.disabled = true;
                                                confirmBtnS.style.background = '#ccc';
                                            }} else {{
                                                coordsElS.textContent = '❌ Need 2 points before confirming';
                                            }}
                                        }});
                                        </script>
                                        """

                                        st.components.v1.html(html_canvas_s, height=700)

                                        # After canvas, input coordinates
                                        col_manual1s, col_manual2s = st.columns(2)

                                        with col_manual1s:
                                            st.write("**Or enter coordinates:**")
                                            col_mx1s, col_my1s = st.columns(2)
                                            with col_mx1s:
                                                mx1s = st.number_input("x1:", min_value=0, max_value=stitched_img.shape[1], key=f"mx1s_{pidx}")
                                            with col_my1s:
                                                my1s = st.number_input("y1:", min_value=0, max_value=stitched_img.shape[0], key=f"my1s_{pidx}")

                                        with col_manual2s:
                                            col_mx2s, col_my2s = st.columns(2)
                                            with col_mx2s:
                                                mx2s = st.number_input("x2:", min_value=0, max_value=stitched_img.shape[1], key=f"mx2s_{pidx}")
                                            with col_my2s:
                                                my2s = st.number_input("y2:", min_value=0, max_value=stitched_img.shape[0], key=f"my2s_{pidx}")

                                        # Use coordinates
                                        final_p1s = (mx1s, my1s)
                                        final_p2s = (mx2s, my2s)

                                        # Calculate only if we have non-zero points
                                        if final_p1s != (0, 0) and final_p2s != (0, 0):
                                            pixel_length_s = np.sqrt((final_p2s[0] - final_p1s[0])**2 + (final_p2s[1] - final_p1s[1])**2)
                                            new_diameter_um_s = pixel_length_s * CALIBRATION_UM_PER_PIXEL

                                            col_ress1, col_ress2 = st.columns(2)
                                            with col_ress1:
                                                st.metric("📐 Line Length", f"{pixel_length_s:.1f} px")
                                            with col_ress2:
                                                st.metric("📏 New Diameter", f"{new_diameter_um_s:.1f} µm")

                                            # RESIZE BUTTON
                                            if st.button(f"✅ APPLY RESIZE", key=f"apply_stitch_{pidx}", use_container_width=True):
                                                push_undo()
                                                st.session_state.results[pidx]["diameter_um"] = new_diameter_um_s
                                                st.session_state.results[pidx]["size_bin"] = get_size_bin(new_diameter_um_s)
                                                st.session_state.results[pidx]["size_method"] = "manual_line_stitched"
                                                st.session_state[f"show_full_{pidx}"] = False
                                                st.success(f"✅ Resized to {new_diameter_um_s:.1f}µm!")
                                                st.rerun()
                                    else:
                                        st.warning("❌ Stitched image not available")
                                except Exception as e:
                                    st.error(f"❌ Could not display stitched image: {str(e)[:100]}")

                        # Normal single-tile view
                        elif filename not in st.session_state.tile_files:
                            st.warning(f"❌ Tile not in upload: {filename}")
                        else:
                            with st.expander(f"Full Image: {filename}", expanded=True):
                                try:
                                    file_obj = st.session_state.tile_files[filename]
                                    tile_img = Image.open(file_obj).convert('RGB')
                                    tile_img = np.array(tile_img)

                                    fig = go.Figure()
                                    fig.add_trace(go.Image(z=tile_img, name="Image"))

                                    # Get box coordinates safely
                                    x = p.get("x", 0)
                                    y = p.get("y", 0)
                                    w = p.get("w", 0)
                                    h = p.get("h", 0)

                                    if x and y and w and h:
                                        fig.add_shape(type="rect", x0=x, y0=y, x1=x+w, y1=y+h,
                                                   line=dict(color="rgb(0, 100, 255)", width=3))

                                    fig.update_layout(
                                        title=f"{filename} | {p.get('class', '?')} ({p.get('size_bin', '?')}) {p.get('diameter_um', '?')}µm",
                                        showlegend=False, hovermode="closest",
                                        margin=dict(b=0, l=0, r=0, t=40), height=600)
                                    fig.update_xaxes(scaleanchor="y", scaleratio=1)
                                    fig.update_yaxes(scaleanchor="x", scaleratio=1)

                                    st.plotly_chart(fig)

                                    c1, c2, c3 = st.columns(3)
                                    with c1:
                                        st.write(f"**Class:** {p.get('class', '?')}")
                                    with c2:
                                        st.write(f"**Size:** {p.get('diameter_um', '?')}µm ({p.get('size_bin', '?')})")
                                    with c3:
                                        st.write(f"**Method:** {p.get('size_method', '?')}")

                                    # LINE MEASUREMENT TOOL - SIMPLE CLICKABLE IMAGE
                                    st.markdown("---")
                                    st.subheader("📏 Resize Particle")
                                    st.write("**Just click on the image twice: first click = red dot, second click = green dot**")

                                    measure_key = f"measure_{pidx}"
                                    if measure_key not in st.session_state:
                                        st.session_state[measure_key] = []

                                    points = st.session_state[measure_key]

                                    # Create canvas with JavaScript click detection
                                    import base64
                                    img_pil = Image.fromarray(tile_img.astype(np.uint8))
                                    buf = io.BytesIO()
                                    img_pil.save(buf, format="PNG")
                                    img_base64 = base64.b64encode(buf.getvalue()).decode()

                                    html_canvas = f"""
                                    <style>
                                    #measureCanvas {{
                                        cursor: crosshair;
                                        border: 3px solid #ccc;
                                        display: block;
                                        max-width: 100%;
                                        height: auto;
                                    }}
                                    #coords {{
                                        font-family: monospace;
                                        margin-top: 10px;
                                        font-size: 14px;
                                        color: #333;
                                    }}
                                    </style>
                                    <canvas id="measureCanvas" width="{tile_img.shape[1]}" height="{tile_img.shape[0]}"></canvas>
                                    <p id="coords">Click image to add points (need 2)</p>
                                    <button id="confirmBtn" style="margin-top: 10px; padding: 10px 20px; font-size: 14px; cursor: pointer; background: #4CAF50; color: white; border: none; border-radius: 4px;">Confirm Points</button>
                                    <script>
                                    const canvas = document.getElementById('measureCanvas');
                                    const ctx = canvas.getContext('2d');
                                    const coordsEl = document.getElementById('coords');
                                    const confirmBtn = document.getElementById('confirmBtn');
                                    const img = new Image();
                                    
                                    let clickPoints = [];
                                    
                                    img.onload = function() {{
                                        ctx.drawImage(img, 0, 0);
                                        redraw();
                                    }};
                                    img.src = 'data:image/png;base64,{img_base64}';
                                    
                                    function redraw() {{
                                        ctx.drawImage(img, 0, 0);
                                        
                                        // Draw existing points
                                        clickPoints.forEach((p, i) => {{
                                            const color = i === 0 ? 'red' : 'green';
                                            ctx.fillStyle = color;
                                            ctx.beginPath();
                                            ctx.arc(p.x, p.y, 12, 0, 2*Math.PI);
                                            ctx.fill();
                                            ctx.strokeStyle = 'white';
                                            ctx.lineWidth = 2;
                                            ctx.stroke();
                                        }});
                                        
                                        // Draw line if 2 points
                                        if (clickPoints.length === 2) {{
                                            ctx.strokeStyle = 'orange';
                                            ctx.lineWidth = 4;
                                            ctx.beginPath();
                                            ctx.moveTo(clickPoints[0].x, clickPoints[0].y);
                                            ctx.lineTo(clickPoints[1].x, clickPoints[1].y);
                                            ctx.stroke();
                                        }}
                                    }}
                                    
                                    canvas.addEventListener('click', function(e) {{
                                        if (clickPoints.length >= 2) return;
                                        
                                        const rect = canvas.getBoundingClientRect();
                                        const x = Math.round((e.clientX - rect.left) * canvas.width / rect.width);
                                        const y = Math.round((e.clientY - rect.top) * canvas.height / rect.height);
                                        
                                        clickPoints.push({{x: x, y: y}});
                                        
                                        if (clickPoints.length === 1) {{
                                            coordsEl.textContent = `Point 1: (${{x}}, ${{y}}) - click again for point 2`;
                                        }} else {{
                                            const dx = clickPoints[1].x - clickPoints[0].x;
                                            const dy = clickPoints[1].y - clickPoints[0].y;
                                            const dist = Math.sqrt(dx*dx + dy*dy);
                                            coordsEl.textContent = `Points ready! Distance: ${{dist.toFixed(1)}}px | Click "Confirm Points" button`;
                                        }}
                                        
                                        redraw();
                                    }});
                                    
                                    confirmBtn.addEventListener('click', function() {{
                                        if (clickPoints.length === 2) {{
                                            window.confirmedPoints_{pidx} = clickPoints;
                                            coordsEl.textContent = `✅ Points confirmed: (${{clickPoints[0].x}}, ${{clickPoints[0].y}}) → (${{clickPoints[1].x}}, ${{clickPoints[1].y}})`;
                                            confirmBtn.disabled = true;
                                            confirmBtn.style.background = '#ccc';
                                        }} else {{
                                            coordsEl.textContent = '❌ Need 2 points before confirming';
                                        }}
                                    }});
                                    </script>
                                    """

                                    st.components.v1.html(html_canvas, height=700)

                                    # After canvas, check if points were confirmed and add them to session state
                                    col_manual1, col_manual2 = st.columns(2)

                                    with col_manual1:
                                        st.write("**Or enter coordinates:**")
                                        col_mx1, col_my1 = st.columns(2)
                                        with col_mx1:
                                            mx1 = st.number_input("x1:", min_value=0, max_value=tile_img.shape[1], key=f"mx1_{pidx}")
                                        with col_my1:
                                            my1 = st.number_input("y1:", min_value=0, max_value=tile_img.shape[0], key=f"my1_{pidx}")

                                    with col_manual2:
                                        col_mx2, col_my2 = st.columns(2)
                                        with col_mx2:
                                            mx2 = st.number_input("x2:", min_value=0, max_value=tile_img.shape[1], key=f"mx2_{pidx}")
                                        with col_my2:
                                            my2 = st.number_input("y2:", min_value=0, max_value=tile_img.shape[0], key=f"my2_{pidx}")

                                    # Use manual inputs
                                    final_p1 = (mx1, my1)
                                    final_p2 = (mx2, my2)

                                    # Calculate only if we have non-zero points
                                    if final_p1 != (0, 0) and final_p2 != (0, 0):
                                        pixel_length = np.sqrt((final_p2[0] - final_p1[0])**2 + (final_p2[1] - final_p1[1])**2)
                                        new_diameter_um = pixel_length * CALIBRATION_UM_PER_PIXEL

                                        col_res1, col_res2 = st.columns(2)
                                        with col_res1:
                                            st.metric("📐 Line Length", f"{pixel_length:.1f} px")
                                        with col_res2:
                                            st.metric("📏 New Diameter", f"{new_diameter_um:.1f} µm")

                                        # RESIZE BUTTON
                                        if st.button(f"✅ APPLY RESIZE", key=f"apply_{pidx}", use_container_width=True):
                                            push_undo()
                                            st.session_state.results[pidx]["diameter_um"] = new_diameter_um
                                            st.session_state.results[pidx]["size_bin"] = get_size_bin(new_diameter_um)
                                            st.session_state.results[pidx]["size_method"] = "manual_line"
                                            st.session_state[f"show_full_{pidx}"] = False
                                            st.success(f"✅ Resized to {new_diameter_um:.1f}µm!")
                                            st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Load error: {str(e)[:60]}")

                    except Exception as e:
                        st.error(f"❌ Unexpected error: {str(e)[:60]}")

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # MASS EDIT
    # ─────────────────────────────────────────────────────────────────────────

    if st.session_state.selected_particles:
        st.subheader("⚙️ Bulk Edit")

        col1, col2 = st.columns(2)
        with col1:
            action = st.radio("Action:", ["Delete", "Change Class"], horizontal=True)
        with col2:
            if action == "Change Class":
                new_cls = st.selectbox("To:", ["Fiber", "Glass", "Metallic", "Other"])

        if st.button("Execute"):
            push_undo()
            for key in st.session_state.selected_particles:
                pidx = int(key.split("_")[1])
                if action == "Delete":
                    st.session_state.results[pidx]["deleted"] = True
                else:
                    st.session_state.results[pidx]["class"] = new_cls

            st.session_state.selected_particles = set()
            st.success(f"✅ Done")
            st.rerun()