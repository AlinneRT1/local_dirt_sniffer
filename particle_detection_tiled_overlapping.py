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

    for idx, tile_meta in enumerate(tile_metadata):
        filename = tile_meta['filename']
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
        draw.rectangle([(x - x1, y - y1), (x + w - x1, y + h - y1)], outline=(0, 100, 255), width=2)

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

        # SIZING BOX - Show measurements
        st.markdown("**📏 Measurements:**")
        st.write(f"Diameter: {p.get('diameter_um', '?'):.1f}µm")
        if p.get("stitched"):
            st.write(f"Original: {p.get('original_diameter_um', '?'):.1f}µm")
            st.write(f"Change: {p.get('size_change_pct', 0):+.1f}%")
        st.write(f"Bbox: {p.get('w', 0)}×{p.get('h', 0)}px")

        # Show duplicate reason if applicable
        if p.get("deleted") and p.get("duplicate_reason"):
            st.info(f"**Why removed:** {p.get('duplicate_reason')}")

        # CHECKBOX - Select particle for mass operations
        key = f"sel_{pidx}"
        is_selected = key in st.session_state.selected_particles
        if st.checkbox("Select", value=is_selected, key=key):
            st.session_state.selected_particles.add(key)
        else:
            st.session_state.selected_particles.discard(key)

        # EDIT CLASS - Change particle classification
        new_cls = st.selectbox(
            "Class:",
            ["Fiber", "Glass", "Metallic", "Other"],
            index=["Fiber", "Glass", "Metallic", "Other"].index(p.get("class", "Other")),
            key=f"cls_{pidx}"
        )
        if new_cls != p.get("class") and st.button("✓", key=f"save_{pidx}"):
            push_undo()
            st.session_state.results[pidx]["class"] = new_cls
            # Recalculate size bin for new class
            st.session_state.results[pidx]["size_bin"] = get_size_bin(
                st.session_state.results[pidx].get("diameter_um", 0))
            st.rerun()

        # DELETE BUTTON
        if st.button("🗑️", key=f"del_{pidx}"):
            push_undo()
            st.session_state.results[pidx]["deleted"] = True
            st.rerun()

        # VIEW FULL BUTTON - Expand to full image
        # Don't show for deleted/duplicate particles (view them in pair comparison instead)
        if not p.get("deleted") and st.button("🔍", key=f"view_{pidx}"):
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

    st.write("**Step 1: Upload manifest.json**")
    manifest_file = st.file_uploader("Manifest:", type=["json"], key="manifest")

    st.write("**Step 2: Upload tile images**")
    tile_files = st.file_uploader(
        "Tile images:",
        type=["jpg", "jpeg", "png", "tif"],
        accept_multiple_files=True,
        key="tiles"
    )

    if manifest_file and tile_files and st.button("📋 Load"):
        try:
            manifest = json.load(manifest_file)
            tile_metadata = manifest.get("tiles", [])

            file_map = {f.name: f for f in tile_files}

            st.session_state.tile_metadata = tile_metadata
            st.session_state.tile_files = file_map

            st.success(f"✅ Ready to detect!")
        except Exception as e:
            st.error(f"Error: {e}")

    st.divider()

    if st.session_state.tile_metadata:
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
                st.write(f"Raw detections: {len(raw_particles)}")

                # Initialize for processing pipeline
                stitched_count = 0
                iou_stats = {'duplicates_removed': 0}

                # SMART EDGE-BASED MATCHING - Compare particles at tile boundaries
                st.write("🔍 Smart edge-based particle matching...")
                try:
                    from intelligent_particle_matcher import IntelligentParticleMatcher

                    # Build tile metadata for matcher
                    matcher_metadata = []
                    for i, tm in enumerate(st.session_state.tile_metadata):
                        matcher_metadata.append({
                            "tile_id": i,
                            "filename": tm["filename"],
                            "width": tm.get("width", 3000),
                            "height": tm.get("height", 3000),
                            "neighbors": tm.get("neighbors", {})
                        })

                    matcher = IntelligentParticleMatcher(
                        matcher_metadata,
                        size_tolerance=0.20,  # 20% size difference ok
                        edge_margin_pct=0.10,  # Check within 10% of edge
                        confidence_threshold=0.5  # Min confidence for match
                    )

                    # Find all edge matches
                    matched_particles, edge_matches = matcher.process_all_neighbors(raw_particles)

                    iou_dedup_count = len([p for p in matched_particles if p.get("deleted")])
                    st.write(f"✅ Smart matching found {len(edge_matches)} matches")
                    st.write(f"   Removed {iou_dedup_count} duplicate particles")

                    # STITCHING: Recalculate sizes on complete stitched particles
                    st.write("🔗 Stitching matched particles...")

                    # Load all tile images for stitching
                    tile_images = {}
                    for tile_idx, tile_meta in enumerate(st.session_state.tile_metadata):
                        filename = tile_meta["filename"]
                        try:
                            tile_file = st.session_state.tile_files[filename]
                            tile_img = Image.open(tile_file).convert('RGB')  # PIL uses RGB, not BGR
                            tile_images[tile_idx] = np.array(tile_img)
                        except Exception as e:
                            st.warning(f"Could not load tile {filename} for stitching: {e}")

                    # Stitch each matched pair
                    stitched_count = 0
                    stitch_results = []
                    marked_merged_count = 0

                    for match in edge_matches:
                        stitch_info = matcher.stitch_particles(
                            match,
                            tile_images,
                            calibration_um_per_pixel=CALIBRATION_UM_PER_PIXEL
                        )

                        if stitch_info:
                            stitched_count += 1
                            stitch_results.append(stitch_info)

                            # Update the kept particle with stitched size
                            kept_particle_idx = match["tile1_idx"]
                            deleted_particle_idx = match["tile2_idx"]

                            if kept_particle_idx < len(matched_particles):
                                kept_p = matched_particles[kept_particle_idx]
                                kept_p["diameter_um"] = stitch_info["stitched_diameter_um"]
                                kept_p["stitched"] = True
                                kept_p["merged"] = True  # Mark as merged for gallery filter
                                kept_p["original_diameter_um"] = stitch_info["original_diameter_um"]
                                kept_p["size_change_pct"] = stitch_info["size_change_pct"]
                                kept_p["stitched_image"] = stitch_info.get("stitched_image")  # Store stitched image
                                kept_p["overlap_pixels"] = stitch_info.get("overlap_pixels", 0)
                                kept_p["stitch_direction"] = stitch_info.get("direction", "?")
                                kept_p["match_score"] = stitch_info.get("match_score", 0)
                                kept_p["stitched_diagonal_px"] = stitch_info.get("stitched_diagonal_px", 0)
                                kept_p["stitched_w"] = stitch_info.get("stitched_w_px", 0)
                                kept_p["stitched_h"] = stitch_info.get("stitched_h_px", 0)

                                # Mark deleted particle and set matched_with
                                if deleted_particle_idx < len(matched_particles):
                                    deleted_p = matched_particles[deleted_particle_idx]
                                    deleted_p["deleted"] = True
                                    deleted_p["matched_with"] = kept_particle_idx
                                    deleted_p["duplicate_type"] = "edge_duplicate"  # Mark as edge duplicate ✅
                                    stitch_direction = stitch_info.get("direction", "?")
                                    deleted_p[
                                        "duplicate_reason"] = f"Cut across {stitch_direction} boundary, kept higher confidence"

                                marked_merged_count += 1

                    st.write(f"✅ Stitched {stitched_count}/{len(edge_matches)} particle pairs")
                    st.write(f"   Marked {marked_merged_count} particles as merged")

                    # DEBUG: Show stitching details
                    with st.expander("🔍 Stitching Debug Info"):
                        st.write("**Stitched Particles Details:**")
                        for idx, stitch in enumerate(stitch_results[:5]):  # Show first 5
                            st.write(f"""
**Pair {idx + 1}:**
- Original: {stitch['original_diameter_um']:.1f}µm
- Stitched: {stitch['stitched_diameter_um']:.1f}µm
- Change: {stitch['size_change_pct']:+.1f}%
- Stitched bbox: {stitch['stitched_w_px']}×{stitch['stitched_h_px']}px
- Overlap: {stitch['overlap_pixels']}px
- Direction: {stitch['direction']}
- Score: {stitch['match_score']:.3f}
                            """)
                        if len(stitch_results) > 5:
                            st.write(f"... and {len(stitch_results) - 5} more")

                    iou_dedup_particles = matched_particles
                    iou_stats = {'duplicates_removed': iou_dedup_count}

                    # Display matched pairs and stitching results
                    if len(edge_matches) > 0:
                        st.info(f"ℹ️ {len(edge_matches)} particle pairs matched at edges, {stitched_count} stitched")
                        with st.expander("📋 Matched particle pairs & stitching results"):
                            for i, match in enumerate(edge_matches):
                                p1 = match["particle1"]
                                p2 = match["particle2"]
                                score = match["match_score"]

                                stitch = stitch_results[i] if i < len(stitch_results) else None

                                if stitch:
                                    st.write(f"""
                                    **Match {i + 1}:** Score {score:.2f}
                                    - **Original:** {p1.get('class')} - {p1.get('diameter_um'):.1f}µm (Tile {match['tile1_id']})
                                    - **Stitched:** {stitch['stitched_diameter_um']:.1f}µm
                                    - **Change:** {stitch['size_change_pct']:+.1f}%
                                    - **Direction:** {match['direction']}
                                    """)
                                else:
                                    st.write(f"""
                                    **Match {i + 1}:** Score {score:.2f}
                                    - {p1.get('class')} ({p1.get('diameter_um'):.1f}µm) 
                                    - {p2.get('class')} ({p2.get('diameter_um'):.1f}µm)
                                    - Direction: {match['direction']}
                                    - ⚠️ Stitching failed
                                    """)

                except Exception as e:
                    st.warning(f"Smart matching failed: {e}")
                    iou_dedup_particles = raw_particles
                # Uses TOLERANCE: particles within 10px of each other = same location
                st.write("🔍 Location deduplication (same spot, different class)...")

                deduplicated_by_location = {}
                location_tolerance = 10  # pixels - particles this close are same location

                for p in iou_dedup_particles:
                    p_center_x = p.get('x', 0) + p.get('w', 0) / 2
                    p_center_y = p.get('y', 0) + p.get('h', 0) / 2
                    tile_id = p.get('tile_id')
                    p_conf = p.get('confidence', 0)

                    # Find if this particle matches any existing location
                    matched = False
                    for existing_key, existing_p in deduplicated_by_location.items():
                        existing_tile_id = existing_key[0]
                        existing_center_x = existing_key[1]
                        existing_center_y = existing_key[2]

                        # Same tile and within tolerance distance?
                        if tile_id == existing_tile_id:
                            dist = ((p_center_x - existing_center_x) ** 2 + (
                                        p_center_y - existing_center_y) ** 2) ** 0.5
                            if dist < location_tolerance:
                                # Same location! Keep higher confidence
                                if p_conf > existing_p.get('confidence', 0):
                                    deduplicated_by_location[existing_key] = p
                                matched = True
                                break

                    if not matched:
                        # New location
                        location_key = (tile_id, p_center_x, p_center_y)
                        deduplicated_by_location[location_key] = p

                # Mark duplicates as deleted and track matches
                location_dedup_count = 0
                kept_particle_ids = set(id(p) for p in deduplicated_by_location.values())
                kept_particles_by_id = {id(p): (idx, p) for idx, p in enumerate(iou_dedup_particles) if
                                        id(p) in kept_particle_ids}

                # Mark deleted and set matched_with
                for idx, p in enumerate(iou_dedup_particles):
                    if id(p) not in kept_particle_ids:
                        p["deleted"] = True

                        # Find which kept particle it matched with
                        p_center_x = p.get('x', 0) + p.get('w', 0) / 2
                        p_center_y = p.get('y', 0) + p.get('h', 0) / 2
                        tile_id = p.get('tile_id')

                        for kept_id, (kept_idx, kept_p) in kept_particles_by_id.items():
                            kept_tile_id = kept_p.get('tile_id')
                            kept_center_x = kept_p.get('x', 0) + kept_p.get('w', 0) / 2
                            kept_center_y = kept_p.get('y', 0) + kept_p.get('h', 0) / 2

                            if tile_id == kept_tile_id:
                                dist = ((p_center_x - kept_center_x) ** 2 + (p_center_y - kept_center_y) ** 2) ** 0.5
                                if dist < location_tolerance:
                                    p["matched_with"] = kept_idx
                                    # Mark as location duplicate (same spot, different class) ✅
                                    p["duplicate_type"] = "location_duplicate"
                                    kept_conf = kept_p.get('confidence', 0)
                                    del_conf = p.get('confidence', 0)
                                    p[
                                        "duplicate_reason"] = f"Same spot: {p.get('class')} ({del_conf:.2f}) vs {kept_p.get('class')} ({kept_conf:.2f})"
                                    break

                        location_dedup_count += 1

                if location_dedup_count > 0:
                    st.warning(
                        f"⚠️ Found {location_dedup_count} duplicate locations (same spot, different class) - kept highest confidence")

                # Debug: count deleted particles
                deleted_in_dedup = len([p for p in iou_dedup_particles if p.get("deleted")])
                st.write(f"   Location dedup marked {deleted_in_dedup} total particles as deleted")

                # After dedup, work with ALL particles (including deleted ones for gallery viewing)
                # Don't filter out deleted particles here - we need them for the "show duplicates only" filter
                all_dedup_particles = iou_dedup_particles  # Keep ALL particles including deleted
                active_particles = [p for p in iou_dedup_particles if not p.get("deleted")]
                st.write(
                    f"After all dedup: {len(active_particles)} active particles (+ {len(iou_dedup_particles) - len(active_particles)} deleted)")

                # Step 2: Mark seams only (disable merging without accurate coords)
                try:
                    st.write("📍 Marking particles at tile seams...")

                    # Mark seams with SIMPLE DIRECT method (tile-local only, no coords needed)
                    # Process ALL particles (including deleted) to keep full record
                    seam_marked = []
                    seams_found = 0
                    seam_margin = 30

                    for p in all_dedup_particles:
                        tile_id = p.get("tile_id", 0)
                        x = p.get("x", 0)
                        y = p.get("y", 0)
                        w = p.get("w", 0)
                        h = p.get("h", 0)

                        # Get tile dimensions from metadata
                        tile_w = 0
                        tile_h = 0
                        if tile_id < len(st.session_state.tile_metadata):
                            tm = st.session_state.tile_metadata[tile_id]
                            tile_w = tm.get("width", 3000)
                            tile_h = tm.get("height", 3000)

                        # Check each edge (tile-local only)
                        seams = []
                        at_seam = False

                        if x < seam_margin:
                            seams.append("left")
                            at_seam = True
                        if x + w > tile_w - seam_margin:
                            seams.append("right")
                            at_seam = True
                        if y < seam_margin:
                            seams.append("top")
                            at_seam = True
                        if y + h > tile_h - seam_margin:
                            seams.append("bottom")
                            at_seam = True

                        p["at_seam"] = at_seam
                        p["seams"] = seams

                        # Only count active particles at seams
                        if at_seam and not p.get("deleted"):
                            seams_found += 1

                        seam_marked.append(p)

                    st.write(f"✅ Found {seams_found} particles at seams")

                    # SKIP MERGING - Without accurate overlap coords, can't reliably match particles
                    # across tile boundaries. Merging would require pixel-perfect alignment.
                    merged_particles = seam_marked
                    merged_pairs = []

                    st.write(f"ℹ️ Merging DISABLED (coords are approximate)")
                    st.write(f"   Seam particles marked but not stitched")

                    num_at_seam = seams_found

                    st.write(f"✅ Processing complete!")
                    st.write(f"   Raw detections: {len(raw_particles)}")
                    st.write(f"   Smart matching removed: {iou_stats['duplicates_removed']}")
                    st.write(f"   Particles stitched: {stitched_count}")
                    st.write(f"   Location dedup removed: {location_dedup_count} (tolerance: 10px)")
                    st.write(f"   Seam particles marked: {num_at_seam}")
                    active_count = len([p for p in merged_particles if not p.get("deleted")])
                    st.write(
                        f"   **Final count (active): {active_count}** (+ {len(merged_particles) - active_count} deleted)")

                    with st.expander("📊 What happened:"):
                        st.write(f"""
                        **Processing Pipeline (Smart Matching):**

                        Raw detections: {len(raw_particles)}
                        ✅ Smart edge matching (size/class/position): -{iou_stats['duplicates_removed']}
                           - Compares particles at tile boundaries
                           - Matches on: size (±20%), class, position alignment
                           - Keeps: highest confidence copy
                        ✅ Location dedup (10px tolerance): -{location_dedup_count}
                           - Removes same particle, different class labels
                        ✅ Seam detection: {num_at_seam} marked
                           - Marks cut particles at edges (not yet stitched)
                        ────────────────────────
                        **FINAL: {len(merged_particles)}**

                        **Matching Criteria:**
                        - Size match: diameter difference ≤ 20%
                        - Class match: same particle class
                        - Position: aligned within tile boundaries
                        - Confidence: at least one detection ≥ 0.5
                        """)

                    with st.expander("ℹ️ Deduplication Details"):
                        st.write("""
                        **Smart Edge-Based Matching (No Blind Assumptions)**

                        Instead of assuming 10% overlap, we use metadata + intelligent criteria:

                        **Step 1: Find Neighbors**
                        - Use metadata "neighbors" field to identify which tiles touch
                        - Check left/right/top/bottom edges

                        **Step 2: Find Edge Particles**
                        - Collect particles within 10% of each tile edge
                        - Prepare to compare across boundaries

                        **Step 3: Smart Matching**
                        For each particle on edge, find best match in neighbor tile:
                        ✅ Size check: diameter difference ≤ 20%
                           - If Particle A is 50µm and Particle B is 58µm → Match (16% diff)
                        ✅ Class check: same type
                           - Both "Fiber" or both "Glass" → Good
                           - Different classes → No match (unless high confidence one)
                        ✅ Position check: aligned geometrically
                           - For left/right edges: Y positions should be similar
                           - For top/bottom edges: X positions should be similar
                        ✅ Confidence check: at least one detection confident
                           - Min 0.5 confidence on highest scorer

                        **Step 4: Keep Highest Confidence**
                        - If all criteria met: particles are likely the SAME particle
                        - Keep the one with higher confidence
                        - Remove the lower confidence duplicate

                        **Step 5: Future - Stitch Cut Particles**
                        - For matched particles at seams: combine images
                        - Recalculate size on stitched image
                        - Update diameter_um with true complete measurement

                        **Example:**
                        ```
                        Tile 0 near right edge: Particle A - 50µm, Fiber, conf 0.92
                        Tile 1 near left edge:  Particle B - 52µm, Fiber, conf 0.78

                        Size: 52-50=2µm, 2/50=4% ✓ (< 20%)
                        Class: Fiber = Fiber ✓
                        Y-pos: Similar ✓
                        Conf: 0.92 ≥ 0.5 ✓

                        → MATCH! Keep A (0.92), delete B (0.78)
                        ```

                        **Why This is Better:**
                        ✅ No blind coordinate assumptions
                        ✅ Uses actual metadata neighbors
                        ✅ Multi-criteria matching (not just overlap)
                        ✅ Keeps high confidence, removes duplicates
                        ✅ Ready for intelligent stitching
                        """)

                    st.session_state.results = merged_particles

                except Exception as e:
                    st.error(f"Seam detection error: {e}")
                    st.write("Using results without seam marks")
                    st.session_state.results = all_dedup_particles

                st.session_state.undo_stack = []
                st.session_state.selected_particles = set()
                st.success(f"Done!")

    st.divider()

    if st.session_state.undo_stack:
        if st.button("↶ Undo"):
            st.session_state.results = st.session_state.undo_stack.pop()
            st.session_state.selected_particles = set()
            st.rerun()

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
                    status = "MERGED (stitched)" if p.get("merged") else (
                        "AT_SEAM (check)" if p.get("at_seam") else "OK")

                    # If merged, try to get recalculated size
                    diameter_um = p["diameter_um"]
                    size_method = p["size_method"]
                    size_bin = p["size_bin"]

                    if p.get("merged"):
                        try:
                            particle_key = f"{p.get('tile_filename')}_{p.get('x')}_{p.get('y')}"

                            if particle_key not in st.session_state.stitch_cache:
                                stitched, merged_meta, seam_info = stitch_merged_particle(st.session_state.tile_files,
                                                                                          p)
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
            if show_duplicates_only:
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

    # If showing duplicates only, pair them with their matches
    if show_duplicates_only:
        # Find duplicate pairs: deleted particle with its kept counterpart
        paired_particles = []
        seen_matches = set()
        unmatched_count = 0

        for idx, p in all_particles:
            if not p.get("deleted"):
                continue

            matched_idx = p.get("matched_with")
            if matched_idx is not None and matched_idx not in seen_matches:
                # Find the kept particle from ALL results (not just filtered all_particles)
                kept_p = None
                if matched_idx < len(st.session_state.results):
                    kept_p = st.session_state.results[matched_idx]

                if kept_p:
                    # Store as pair: (kept, deleted)
                    paired_particles.append(((matched_idx, kept_p), (idx, p)))
                    seen_matches.add(matched_idx)
            else:
                # No matched_with set
                unmatched_count += 1

        # Debug info
        st.info(
            f"✅ Found {len(paired_particles)} duplicate pairs | ❌ {unmatched_count} deleted particles without match link")

        display_particles = paired_particles
    else:
        # Regular display (not paired)
        display_particles = all_particles

    # Determine if showing pairs BEFORE using in pagination
    is_showing_pairs = show_duplicates_only and len(display_particles) > 0 and isinstance(display_particles[0],
                                                                                          tuple) and isinstance(
        display_particles[0][0], tuple)

    if all_particles:
        # Pagination - calculate based on what we're displaying
        if is_showing_pairs:
            # For pairs: pairs_per_page pairs = items_per_page items (2 particles per pair)
            pairs_per_page = max(1, items_per_page // 2)
            total_pairs = len(display_particles)
            total_pages = max(1, (total_pairs + pairs_per_page - 1) // pairs_per_page)

            st.write(f"Showing {total_pairs} duplicate pairs ({total_pairs * 2} items)")
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

                                        st.markdown(
                                            f"**🔴 Red line at x≈{p.get('boundary_line_pos', '?')}px = tile boundary | 🔵 Blue box = particle extent**")

                                        # SIZING BOX - Show measurements
                                        st.markdown("---")
                                        st.markdown("### 📐 Measurements & Pixel Details")

                                        col1, col2, col3, col4 = st.columns(4)
                                        with col1:
                                            st.metric("Original Diameter",
                                                      f"{p.get('original_diameter_um', '?'):.1f}µm")
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
                                            st.write(
                                                f"**Stitched Bbox:** {p.get('stitched_w', 0)}×{p.get('stitched_h', 0)}px")
                                        with col3:
                                            original_diag = np.sqrt(p.get('w', 0) ** 2 + p.get('h', 0) ** 2) if p.get(
                                                'w', 0) > 0 else 0
                                            st.write(
                                                f"**Diagonal:** {original_diag:.0f}px → {p.get('stitched_diagonal_px', 0):.0f}px")

                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.write(f"**Overlap:** {p.get('overlap_pixels', 0)}px")
                                        with col2:
                                            st.write(f"**Direction:** {p.get('stitch_direction', '?')}")
                                        with col3:
                                            st.write(f"**Class:** {p.get('class', '?')} | {p.get('size_bin', '?')}")
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
                                        fig.add_shape(type="rect", x0=x, y0=y, x1=x + w, y1=y + h,
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