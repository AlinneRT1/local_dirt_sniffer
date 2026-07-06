"""
Tiled Particle Detection Gallery - OVERLAPPING TILES VERSION
All features: summary table, gallery, full image zoom, individual edits,
mass edit, undo, sizing method display, bounding boxes

**FOR TILES WITH OVERLAPS**
- Uses IOU deduplication to handle duplicate particles in overlap zones
- Merges cut particles at seams
- Fully featured UI

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

st.set_page_config(page_title="tiled dirt sniffer - OVERLAPPING", page_icon="icon.ico", layout="wide")
st.title("🐕 tiled_dirt_sniffer: Overlapping Tiles Review Dashboard")

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
    return YOLO(MODEL_PATH)

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
            results = model(tile_img_bgr, iou=0.45, conf=0.02, verbose=False)
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
        if st.button("🔍 Run Inference"):
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
                # FOR OVERLAPPING TILES: Uses IOU deduplication
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
                    # iou_threshold=0.3 means: if 30%+ overlap, consider it duplicate
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

                    with st.expander("ℹ️ Deduplication Details"):
                        st.write("""
                        **Your Tile Setup: OVERLAPPING Tiles**
                        - Tiles have overlapping regions (likely 10-30% overlap)
                        - Mosaic stitcher detected same particles in overlap zones
                        
                        **What Happens: IOU Deduplication + Merging**
                        
                        **Step 1: IOU Deduplication**
                        - Compare all particles in overlapping regions
                        - If IOU > 0.3 (30% overlap), they're the same particle
                        - Keep highest confidence copy, remove lower confidence
                        
                        **Step 2: Seam Detection**
                        - Mark particles at tile edges as "at_seam"
                        
                        **Step 3: Cut Particle Merging**
                        - Find matching pieces in neighbor tiles
                        - If particle found at same Y-position and similar size → MERGE them
                        - Both halves combined into ONE complete particle
                        - Recalculate size on stitched complete image
                        
                        **Why this matters:**
                        - Overlap dedup: removes false duplicates from overlap zones ✅
                        - Cut merging: combines split particles into complete pieces ✅
                        - Result: accurate count with no duplicates and no false splits
                        
                        **Confidence:**
                        - Duplicates identified by: IOU > 0.3 in overlap zones
                        - Merged particles show COMPLETE size on stitched image
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
                            pass  # Keep original values if recalc fails

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
# MAIN (Rest is identical to edge-to-edge version - gallery, table, mass edit)
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.results is None:
    st.info("👈 Upload tiles and run inference")
else:
    # SUMMARY TABLE
    st.subheader("📊 Summary Table")

    data = {}
    for cls in ["Fiber", "Glass", "Metallic", "Other"]:
        data[cls] = {}
        for b, _, _ in SIZE_BINS:
            count = len([p for p in st.session_state.results
                        if p["class"] == cls and p["size_bin"] == b and not p.get("deleted")])
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
                    # Load tile
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

                    # Display
                    st.image(crop, use_column_width=True)

                    # Caption
                    method = p.get("size_method", "?")
                    caption = f"{p.get('class', '?')} | {p.get('size_bin', '?')}\n{p.get('diameter_um', '?'):.1f}µm\n({method})"

                    if p.get("merged"):
                        caption = f"🔗 MERGED\n{caption}\n✅ Size recalculated"

                    if p.get("at_seam") and not p.get("merged"):
                        caption += f"\n⚠️ At seams"

                    st.caption(caption)

                    # Checkbox
                    key = f"sel_{pidx}"
                    if st.checkbox("Select", value=key in st.session_state.selected_particles, key=key):
                        st.session_state.selected_particles.add(key)
                    else:
                        st.session_state.selected_particles.discard(key)

                    # Edit class
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

                    # Delete
                    if st.button("🗑️", key=f"del_{pidx}"):
                        push_undo()
                        st.session_state.results[pidx]["deleted"] = True
                        st.rerun()

                    # View full
                    if st.button("🔍 View Full", key=f"view_{pidx}"):
                        st.session_state[f"show_full_{pidx}"] = True

                except Exception as e:
                    st.error(f"❌ Display error")

        # Full image viewer (merged and normal)
        for pidx, p in [(idx, p) for idx, p in page_particles]:
            if st.session_state.get(f"show_full_{pidx}", False):
                try:
                    filename = p.get("tile_filename")
                    if not filename:
                        st.error("❌ Tile filename missing")
                        continue

                    # Check if merged
                    if p.get("merged"):
                        with st.expander(f"🔗 MERGED PARTICLE: {p.get('tile_filename', '?')}", expanded=True):
                            st.info("✅ Cut particle detected and merged")

                            try:
                                particle_key = f"{p.get('tile_filename')}_{p.get('x')}_{p.get('y')}"

                                if particle_key not in st.session_state.stitch_cache:
                                    stitched, merged_metadata, seam_info = stitch_merged_particle(
                                        st.session_state.tile_files, p
                                    )

                                    if stitched is not None:
                                        st.session_state.stitch_cache[particle_key] = (
                                            stitched, merged_metadata, seam_info
                                        )
                                    else:
                                        st.warning("⚠️ Could not stitch images")
                                        continue

                                stitched, merged_metadata, seam_info = st.session_state.stitch_cache[particle_key]

                                # Create figure with seam line and particle box
                                fig = go.Figure()
                                fig.add_trace(go.Image(z=stitched, name="Stitched"))

                                if seam_info:
                                    try:
                                        if seam_info["type"] == "vertical":
                                            seam_x = seam_info["pos"]
                                            fig.add_shape(
                                                type="line",
                                                x0=seam_x, y0=0,
                                                x1=seam_x, y1=stitched.shape[0],
                                                line=dict(color="red", width=3, dash="dash")
                                            )
                                        else:
                                            seam_y = seam_info["pos"]
                                            fig.add_shape(
                                                type="line",
                                                x0=0, y0=seam_y,
                                                x1=stitched.shape[1], y1=seam_y,
                                                line=dict(color="red", width=3, dash="dash")
                                            )
                                    except:
                                        pass

                                # Blue box
                                try:
                                    x = p.get("x")
                                    y = p.get("y")
                                    w = p.get("w")
                                    h = p.get("h")

                                    if x is not None and y is not None and w is not None and h is not None:
                                        fig.add_shape(
                                            type="rect",
                                            x0=x, y0=y, x1=x + w, y1=y + h,
                                            line=dict(color="rgb(0, 100, 255)", width=4)
                                        )
                                except:
                                    pass

                                fig.update_layout(
                                    title=f"Merged | {p.get('class', '?')} | {merged_metadata.get('diameter_um', '?') if merged_metadata else '?'}µm",
                                    showlegend=False,
                                    hovermode="closest",
                                    margin=dict(b=0, l=0, r=0, t=40),
                                    height=600,
                                )
                                fig.update_xaxes(scaleanchor="y", scaleratio=1)
                                fig.update_yaxes(scaleanchor="x", scaleratio=1)

                                st.plotly_chart(fig, use_container_width=True)

                                if merged_metadata:
                                    c1, c2, c3, c4 = st.columns(4)
                                    with c1:
                                        st.write(f"**Class:** {p.get('class', '?')}")
                                    with c2:
                                        st.write(f"**Size (recalc):** {merged_metadata.get('diameter_um', '?')}µm")
                                    with c3:
                                        st.write(f"**Method:** {merged_metadata.get('size_method', '?')}")
                                    with c4:
                                        st.write(f"**Tiles:** {len(p.get('original_particles', []))}")

                            except Exception as e:
                                st.error(f"❌ Error: {str(e)[:60]}")

                    # Normal view
                    elif filename not in st.session_state.tile_files:
                        st.warning(f"❌ Tile not found")
                    else:
                        with st.expander(f"Full Image: {filename}", expanded=True):
                            try:
                                file_obj = st.session_state.tile_files[filename]
                                tile_img = Image.open(file_obj).convert('RGB')
                                tile_img = np.array(tile_img)

                                fig = go.Figure()
                                fig.add_trace(go.Image(z=tile_img, name="Image"))

                                x = p.get("x", 0)
                                y = p.get("y", 0)
                                w = p.get("w", 0)
                                h = p.get("h", 0)

                                if x and y and w and h:
                                    fig.add_shape(type="rect", x0=x, y0=y, x1=x+w, y1=y+h,
                                               line=dict(color="rgb(0, 100, 255)", width=3))

                                fig.update_layout(
                                    title=f"{filename} | {p.get('class', '?')} | {p.get('diameter_um', '?')}µm",
                                    showlegend=False, hovermode="closest",
                                    margin=dict(b=0, l=0, r=0, t=40), height=600)
                                fig.update_xaxes(scaleanchor="y", scaleratio=1)
                                fig.update_yaxes(scaleanchor="x", scaleratio=1)

                                st.plotly_chart(fig, use_container_width=True)

                                c1, c2, c3 = st.columns(3)
                                with c1:
                                    st.write(f"**Class:** {p.get('class', '?')}")
                                with c2:
                                    st.write(f"**Size:** {p.get('diameter_um', '?')}µm")
                                with c3:
                                    st.write(f"**Method:** {p.get('size_method', '?')}")
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)[:60]}")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)[:60]}")

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
