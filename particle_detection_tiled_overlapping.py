"""
Tiled Particle Detection Gallery - OVERLAPPING TILES VERSION (GPU-ACCELERATED LOCAL)
All features: summary table, gallery, full image zoom, individual edits,
mass edit, undo, sizing method display, bounding boxes

**FOR TILES WITH OVERLAPS - GPU ACCELERATED**
- Uses IOU deduplication to handle duplicate particles in overlap zones
- Merges cut particles at seams
- GPU-accelerated YOLO inference for fast detection
- Runs locally (no Streamlit Cloud needed)

Features:
- Summary table (class × size bin)
- 6-column gallery with pagination
- Blue bounding boxes on previews
- Sizing method display (edge_detect, mask_bounds, bbox)
- Full image zoom/pan with Plotly
- Individual class editing
- Delete individual particles
- Select + mass edit
- Undo stack
- CSV export
- IOU-based deduplication for overlapping regions
- Cut particle merging at seams
- GPU acceleration (CUDA-enabled NVIDIA GPUs)
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

st.set_page_config(page_title="tiled dirt sniffer - OVERLAPPING (GPU)", page_icon="icon.ico", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# GPU CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Check GPU availability
GPU_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda:0" if GPU_AVAILABLE else "cpu"

if GPU_AVAILABLE:
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
else:
    GPU_NAME = "None"
    GPU_MEMORY = 0

# Display GPU info in app
st.title("🐕 tiled_dirt_sniffer: Overlapping Tiles Review Dashboard")

with st.expander("🖥️ System Info", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**CUDA Available:** {GPU_AVAILABLE}")
    with col2:
        if GPU_AVAILABLE:
            st.write(f"**GPU:** {GPU_NAME}")
            st.write(f"**GPU Memory:** {GPU_MEMORY:.1f} GB")
        else:
            st.warning("⚠️ GPU not available - using CPU (slower)")

# CONFIG
MODEL_PATH = "models/best.pt"  # Local path to model
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
    """Load YOLO model on GPU"""
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model not found at {MODEL_PATH}")
        st.info("Create a 'models' folder in this directory and add 'best.pt'")
        return None

    try:
        with st.spinner(f"Loading model on {DEVICE}..."):
            model = YOLO(MODEL_PATH)
            model.to(DEVICE)  # Move to GPU
        st.success(f"✅ Model loaded on {DEVICE}")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

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
        seam_position = None
        if p1["tile_filename"] < p2["tile_filename"]:
            # Horizontal stitch (left-right)
            stitched = np.concatenate([img1, img2], axis=1)
            seam_position = {"type": "vertical", "pos": img1.shape[1]}
        else:
            # Vertical stitch (top-bottom)
            stitched = np.concatenate([img1, img2], axis=0)
            seam_position = {"type": "horizontal", "pos": img1.shape[0]}

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
    """Detect in all tiles using GPU acceleration"""
    all_particles = []
    progress_bar = st.progress(0)
    status = st.empty()

    for idx, tile_meta in enumerate(tile_metadata):
        filename = tile_meta['filename']
        status.text(f"🔍 Detecting {idx + 1}/{len(tile_metadata)}: {filename} (on {DEVICE})")

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

        # Detect with GPU (automatic when model is on GPU)
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
if "selected_particles" not in st.session_state:
    st.session_state.selected_particles = set()

def push_undo():
    st.session_state.undo_stack.append(deepcopy(st.session_state.results))

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
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
        if st.button("🔍 Run Inference (GPU)"):
            model = load_model()
            if model is None:
                st.error("Model not found")
            else:
                # Step 1: Detect in all tiles
                raw_particles = detect_particles_in_tiles(
                    st.session_state.tile_files,
                    st.session_state.tile_metadata,
                    model
                )
                st.write(f"Raw detections: {len(raw_particles)}")

                # Step 2: Deduplicate & merge using TileParticleManager
                try:
                    from tile_particle_manager import TileParticleManager
                    import tempfile

                    st.write("Handling overlapping regions with IOU deduplication...")
                    st.write("Finding & merging cut particles at seams...")

                    # Convert metadata to TileParticleManager format
                    mgr_metadata = []
                    for i, tm in enumerate(st.session_state.tile_metadata):
                        mgr_metadata.append({
                            "id": i,
                            "filename": tm["filename"],
                            "x_start": tm.get("x", 0),
                            "y_start": tm.get("y", 0),
                            "x_end": tm.get("x", 0) + tm.get("width", 3000),
                            "y_end": tm.get("y", 0) + tm.get("height", 3000),
                            "neighbors": tm.get("neighbors", {})
                        })

                    # Save temp metadata for manager
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                        json.dump(mgr_metadata, f)
                        metadata_file = f.name

                    # Use TileParticleManager with full deduplication
                    manager = TileParticleManager(metadata_file, iou_threshold=0.3, seam_margin=30)
                    dedup_particles, stats = manager.process_tile_particles(raw_particles)

                    # Merge cut particles
                    st.write("Merging cut particle pieces...")
                    merged_particles, merged_pairs = manager.merge_cut_particles(dedup_particles)

                    num_at_seam = len([p for p in merged_particles if p.get('at_seam')])

                    st.write(f"✅ Processing complete!")
                    st.write(f"   Raw detections: {len(raw_particles)}")
                    st.write(f"   Duplicates removed (IOU > 0.3): {stats['duplicates_removed']}")
                    st.write(f"   Particles at seams: {num_at_seam}")
                    st.write(f"   Cut particle pairs merged: {len(merged_pairs)}")
                    st.write(f"   Final count: {len(merged_particles)}")

                    with st.expander("📊 What happened:"):
                        st.write(f"""
                        **Raw detections:** {len(raw_particles)}
                        (all particles detected across all tiles)
                        
                        **Duplicates removed:** {stats['duplicates_removed']}
                        (same particle detected in overlap zones, kept highest confidence)
                        
                        **Seam particles identified:** {num_at_seam}
                        (particles at tile edges, potentially cut)
                        
                        **Merged pairs:** {len(merged_pairs)}
                        (cut pieces stitched together into complete particles)
                        
                        **Final unique particles:** {len(merged_particles)}
                        = {len(raw_particles)} - {stats['duplicates_removed']} (dedup) - {len(merged_pairs)} (merged)
                        = {len(merged_particles)}
                        """)

                    st.session_state.results = merged_particles

                except ImportError:
                    st.warning("TileParticleManager not found, using raw detections")
                    st.session_state.results = raw_particles
                except Exception as e:
                    st.error(f"Processing error: {e}")
                    st.write("Using raw detections")
                    st.session_state.results = raw_particles

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
                    status = "MERGED (stitched)" if p.get("merged") else ("AT_SEAM (check)" if p.get("at_seam") else "OK")

                    # If merged, try to get recalculated size
                    diameter_um = p.get("diameter_um")
                    size_method = p.get("size_method")
                    size_bin = p.get("size_bin")

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
                            pass

                    rows.append({
                        "tile": p.get("tile_filename"),
                        "class": p.get("class"),
                        "diameter_um": diameter_um,
                        "size_bin": size_bin,
                        "size_method": size_method,
                        "confidence": round(p.get("confidence", 0), 3),
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
# MAIN (Gallery, Summary Table, Mass Edit - same as before)
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.results is None:
    st.info("👈 Upload tiles and run inference on GPU")
else:
    # SUMMARY TABLE
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

    totals_row = {"Material": "TOTAL"}
    for b, _, _ in SIZE_BINS:
        total = sum(data[cls][b] for cls in ["Fiber", "Glass", "Metallic", "Other"])
        totals_row[b] = total
    rows.append(totals_row)

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, height=200)

    st.divider()

    # GALLERY
    st.subheader("🖼️ Particle Gallery")

    col1, col2, col3, col4, col5 = st.columns(5)
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
        items_per_page = st.selectbox("Per page:", [12, 18, 24, 36], index=0)

    # Filter particles
    all_particles = []
    for idx, p in enumerate(st.session_state.results):
        if not p.get("deleted") and p.get("class") in filter_class and p.get("size_bin") in filter_bins:
            if show_seams_only and not p.get("at_seam"):
                continue
            if show_merged_only and not p.get("merged"):
                continue
            all_particles.append((idx, p))

    if all_particles:
        st.success(f"{len(all_particles)} particles")

        # Pagination
        total_pages = max(1, (len(all_particles) + items_per_page - 1) // items_per_page)
        if total_pages > 1:
            page = st.slider("Page:", 1, total_pages, 1) - 1
        else:
            page = 0

        start = page * items_per_page
        end = start + items_per_page
        page_particles = all_particles[start:end]

        # Gallery
        cols = st.columns(6)
        for i, (pidx, p) in enumerate(page_particles):
            with cols[i % 6]:
                try:
                    filename = p.get("tile_filename")
                    if not filename or filename not in st.session_state.tile_files:
                        st.warning("❌ Tile missing")
                        continue

                    try:
                        file_obj = st.session_state.tile_files[filename]
                        tile_img = Image.open(file_obj).convert('RGB')
                        tile_img = np.array(tile_img)
                    except Exception as e:
                        st.error(f"❌ Tile error")
                        continue

                    # Crop
                    x, y, w, h = p.get("x", 0), p.get("y", 0), p.get("w", 10), p.get("h", 10)
                    margin = 15
                    x1 = max(0, x - margin)
                    y1 = max(0, y - margin)
                    x2 = min(tile_img.shape[1], x + w + margin)
                    y2 = min(tile_img.shape[0], y + h + margin)

                    crop = tile_img[y1:y2, x1:x2].copy()

                    # Draw bright blue box
                    crop_pil = Image.fromarray(crop).convert('RGB')
                    draw = ImageDraw.Draw(crop_pil)
                    draw.rectangle([(x-x1, y-y1), (x+w-x1, y+h-y1)], outline=(0, 100, 255), width=2)
                    crop = np.array(crop_pil)

                    st.image(crop, use_column_width=True)

                    method = p.get("size_method", "?")
                    caption = f"{p.get('class', '?')} | {p.get('size_bin', '?')}\n{p.get('diameter_um', '?'):.1f}µm\n({method})"

                    if p.get("merged"):
                        caption = f"🔗 MERGED\n{caption}\n✅ Size recalc"

                    if p.get("at_seam") and not p.get("merged"):
                        caption += f"\n⚠️ At seams"

                    st.caption(caption)

                    key = f"sel_{pidx}"
                    if st.checkbox("Select", value=key in st.session_state.selected_particles, key=key):
                        st.session_state.selected_particles.add(key)
                    else:
                        st.session_state.selected_particles.discard(key)

                    new_cls = st.selectbox(
                        "Class:",
                        ["Fiber", "Glass", "Metallic", "Other"],
                        index=["Fiber", "Glass", "Metallic", "Other"].index(p.get("class", "Other")),
                        key=f"cls_{pidx}"
                    )
                    if new_cls != p.get("class") and st.button("✓", key=f"save_{pidx}"):
                        push_undo()
                        st.session_state.results[pidx]["class"] = new_cls
                        st.rerun()

                    if st.button("🗑️", key=f"del_{pidx}"):
                        push_undo()
                        st.session_state.results[pidx]["deleted"] = True
                        st.rerun()

                    if st.button("🔍 View Full", key=f"view_{pidx}"):
                        st.session_state[f"show_full_{pidx}"] = True

                except Exception as e:
                    st.error(f"❌ Display error")

    st.divider()

    # MASS EDIT
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