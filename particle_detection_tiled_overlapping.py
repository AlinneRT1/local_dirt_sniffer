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

st.set_page_config(page_title="tiled dirt sniffer", page_icon="icon.ico", layout="wide")
st.title("🐕 tiled_dirt_sniffer: Review Dashboard")

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
                    for tile_meta in st.session_state.tile_metadata:
                        tile_idx = tile_meta["tile_id"]
                        filename = tile_meta["filename"]
                        try:
                            tile_file = st.session_state.tile_files[filename]
                            tile_img = Image.open(tile_file).convert('BGR')
                            tile_images[tile_idx] = np.array(tile_img)
                        except Exception as e:
                            st.warning(f"Could not load tile {filename} for stitching: {e}")

                    # Stitch each matched pair
                    stitched_count = 0
                    stitch_results = []

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
                            if kept_particle_idx < len(matched_particles):
                                kept_p = matched_particles[kept_particle_idx]
                                kept_p["diameter_um"] = stitch_info["stitched_diameter_um"]
                                kept_p["stitched"] = True
                                kept_p["original_diameter_um"] = stitch_info["original_diameter_um"]
                                kept_p["size_change_pct"] = stitch_info["size_change_pct"]

                    st.write(f"✅ Stitched {stitched_count}/{len(edge_matches)} particle pairs")

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

                # Mark duplicates as deleted
                location_dedup_count = 0
                kept_particles = set(id(p) for p in deduplicated_by_location.values())
                for p in iou_dedup_particles:
                    if id(p) not in kept_particles:
                        p["deleted"] = True
                        location_dedup_count += 1

                if location_dedup_count > 0:
                    st.warning(
                        f"⚠️ Found {location_dedup_count} duplicate locations (same spot, different class) - kept highest confidence")

                # After dedup, work with non-deleted particles
                dedup_particles = [p for p in iou_dedup_particles if not p.get("deleted")]
                st.write(f"After all dedup: {len(dedup_particles)} particles")

                # Step 2: Mark seams only (disable merging without accurate coords)
                try:
                    st.write("📍 Marking particles at tile seams...")

                    # Mark seams with SIMPLE DIRECT method (tile-local only, no coords needed)
                    seam_marked = []
                    seams_found = 0
                    seam_margin = 30

                    for p in dedup_particles:
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

                        if at_seam:
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
                    st.write(f"   **Final count: {len(merged_particles)}**")

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
                    st.session_state.results = dedup_particles

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
    st.dataframe(df, use_container_width=True, height=200)

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

    # Filter particles
    all_particles = []
    if st.session_state.results is not None:
        for idx, p in enumerate(st.session_state.results):
            if not p.get("deleted") and p.get("class") in filter_class and p.get("size_bin") in filter_bins:
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
                    draw.rectangle([(x - x1, y - y1), (x + w - x1, y + h - y1)], outline=(0, 100, 255), width=2)
                    crop = np.array(crop_pil)

                    # Resize to constant width (250px) while maintaining aspect ratio
                    crop_pil = Image.fromarray(crop).convert('RGB')
                    aspect_ratio = crop_pil.height / crop_pil.width
                    new_height = int(250 * aspect_ratio)
                    crop_pil = crop_pil.resize((250, new_height), Image.Resampling.LANCZOS)

                    # Display resized image
                    st.image(crop_pil)

                    # Caption with size bin and sizing method
                    method = p.get("size_method", "?")
                    caption = f"{p.get('class', '?')} | {p.get('size_bin', '?')}\n{p.get('diameter_um', '?'):.1f}µm ({method})\nConf: {p.get('confidence', 0):.2f}"

                    # Mark merged particles
                    if p.get("merged"):
                        caption = f"🔗 MERGED\n{caption}\n✅ Size recalculated"

                    # Add seam warning if applicable
                    if p.get("at_seam") and not p.get("merged"):
                        caption += f"\n⚠️ At seams"

                    st.caption(caption)

                    # Checkbox
                    key = f"sel_{pidx}"
                    is_selected = key in st.session_state.selected_particles
                    if st.checkbox("Select", value=is_selected, key=key):
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

        # Full image viewer
        for pidx, p in [(idx, p) for idx, p in page_particles]:
            if st.session_state.get(f"show_full_{pidx}", False):
                try:
                    filename = p.get("tile_filename")
                    if not filename:
                        st.error("❌ Tile filename missing")
                        continue

                    # Check if merged - show stitched image
                    if p.get("merged"):
                        with st.expander(f"🔗 MERGED PARTICLE: {p.get('tile_filename', '?')}", expanded=True):
                            st.info("✅ Cut particle detected and merged")

                            try:
                                # Use cache to avoid re-stitching
                                particle_key = f"{p.get('tile_filename')}_{p.get('x')}_{p.get('y')}"

                                if particle_key not in st.session_state.stitch_cache:
                                    # First time: stitch and cache
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

                                # Retrieve from cache
                                stitched, merged_metadata, seam_info = st.session_state.stitch_cache[particle_key]

                                # Create Plotly figure
                                fig = go.Figure()
                                fig.add_trace(go.Image(z=stitched, name="Stitched"))

                                # Draw RED SEAM LINE
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
                                            fig.add_annotation(
                                                x=seam_x, y=10,
                                                text="SEAM",
                                                showarrow=False,
                                                font=dict(color="red", size=12),
                                                bgcolor="white"
                                            )
                                        else:
                                            seam_y = seam_info["pos"]
                                            fig.add_shape(
                                                type="line",
                                                x0=0, y0=seam_y,
                                                x1=stitched.shape[1], y1=seam_y,
                                                line=dict(color="red", width=3, dash="dash")
                                            )
                                            fig.add_annotation(
                                                x=10, y=seam_y,
                                                text="SEAM",
                                                showarrow=False,
                                                font=dict(color="red", size=12),
                                                bgcolor="white"
                                            )
                                    except Exception as e:
                                        st.warning(f"⚠️ Could not draw seam line: {str(e)[:40]}")

                                # Draw BLUE BOX around merged particle
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
                                except Exception as e:
                                    st.warning(f"⚠️ Could not draw particle box: {str(e)[:40]}")

                                # Update layout
                                try:
                                    fig.update_layout(
                                        title=f"Merged Particle (Stitched) | {p.get('class', '?')} | {merged_metadata.get('diameter_um', '?') if merged_metadata else '?'}µm",
                                        showlegend=False,
                                        hovermode="closest",
                                        margin=dict(b=0, l=0, r=0, t=40),
                                        height=600,
                                    )
                                    fig.update_xaxes(scaleanchor="y", scaleratio=1)
                                    fig.update_yaxes(scaleanchor="x", scaleratio=1)

                                    st.plotly_chart(fig, use_container_width=True)
                                except Exception as e:
                                    st.error(f"❌ Chart error: {str(e)[:40]}")

                                # Display metadata
                                if merged_metadata:
                                    try:
                                        c1, c2, c3, c4 = st.columns(4)
                                        with c1:
                                            st.write(f"**Class:** {p.get('class', '?')}")
                                        with c2:
                                            st.write(
                                                f"**Size (recalc):** {merged_metadata.get('diameter_um', '?')}µm ({merged_metadata.get('size_bin', '?')})")
                                        with c3:
                                            st.write(f"**Method:** {merged_metadata.get('size_method', '?')}")
                                        with c4:
                                            st.write(f"**Tiles:** {len(p.get('original_particles', []))}")
                                    except Exception as e:
                                        st.warning(f"⚠️ Could not display metadata: {str(e)[:40]}")

                            except Exception as e:
                                st.error(f"❌ Merged view error: {str(e)[:60]}")

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

                                st.plotly_chart(fig, use_container_width=True)

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