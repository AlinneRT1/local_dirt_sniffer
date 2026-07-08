"""
FLEXIBLE PARTICLE PROCESSING PIPELINE

Handles three scenarios:
1. RAW TILES (no metadata) - just over-labeling detection, no stitching
2. TILED (with neighbor metadata) - IOU dedup + over-labeling + stitching
3. STITCHED (metadata indicates stitched) - skip edge dedup, handle cuts + over-labeling
"""

import streamlit as st
from intelligent_particle_matcher import IntelligentParticleMatcher


def detect_data_type(tile_metadata):
    """
    Detect which type of data we're working with

    Returns: 'raw', 'tiled', or 'stitched'
    """
    if not tile_metadata:
        return 'raw'

    sample = tile_metadata[0] if isinstance(tile_metadata, list) else tile_metadata

    # Check if stitched flag exists
    if sample.get('stitched') or sample.get('is_stitched'):
        return 'stitched'

    # Check if has row/col (tiled indicator)
    if 'row' in sample or 'col' in sample:
        return 'tiled'

    # Check if has bbox_in_mosaic (positional data)
    if 'bbox_in_mosaic' in sample:
        return 'tiled'

    # No metadata = raw
    return 'raw'


def run_full_dedup_and_stitch_pipeline(raw_particles, tile_metadata, tile_files):
    """
    Complete flexible pipeline that handles three data types

    Returns:
        (final_particles, stats_dict)
    """

    stats = {
        'raw': len(raw_particles),
        'data_type': 'unknown',
        'after_iou_dedup': 0,
        'after_overlabel_dedup': 0,
        'stitched_pairs': 0,
        'final': 0,
    }

    # ============================================================================
    # DETECT DATA TYPE
    # ============================================================================
    data_type = detect_data_type(tile_metadata)
    stats['data_type'] = data_type

    st.write(f"### 📊 Data Type: **{data_type.upper()}**")

    if data_type == 'raw':
        return handle_raw_tiles(raw_particles, stats)
    elif data_type == 'tiled':
        return handle_tiled_with_neighbors(raw_particles, tile_metadata, tile_files, stats)
    elif data_type == 'stitched':
        return handle_stitched_images(raw_particles, tile_metadata, tile_files, stats)

    return raw_particles, stats


# ============================================================================
# SCENARIO 1: RAW TILES (NO METADATA)
# ============================================================================

def handle_raw_tiles(raw_particles, stats):
    """
    Raw tiles with no metadata:
    - Just do over-labeling detection
    - No edge detection or stitching
    - Output simple report
    """
    st.write("**Mode: Raw tiles (no stitching, only over-labeling)**")

    particles = raw_particles.copy()

    # Only over-labeling: same spot, different class
    st.write("**Step 1: Over-labeling detection**")

    overlabel_removed = 0

    for i, p1 in enumerate(particles):
        if p1.get('deleted'):
            continue

        for j, p2 in enumerate(particles):
            if j <= i or p2.get('deleted'):
                continue

            # Only check particles on SAME tile (if tile_id exists)
            if p1.get('tile_id') != p2.get('tile_id'):
                # Or if no tile_id, check all particles
                if 'tile_id' in p1 and 'tile_id' in p2:
                    continue

            # Check if same location
            p1_x = p1.get('x', 0) + p1.get('w', 0) / 2
            p1_y = p1.get('y', 0) + p1.get('h', 0) / 2
            p2_x = p2.get('x', 0) + p2.get('w', 0) / 2
            p2_y = p2.get('y', 0) + p2.get('h', 0) / 2

            dist = ((p1_x - p2_x)**2 + (p1_y - p2_y)**2)**0.5

            if dist < 20:  # Within 20px = same spot
                # Different class?
                if p1.get('class') != p2.get('class'):
                    # Keep higher confidence
                    if p1['confidence'] > p2['confidence']:
                        particles[j]['deleted'] = True
                        particles[j]['duplicate_type'] = 'location_duplicate'
                        particles[j]['matched_with'] = i  # Link to kept particle
                        particles[j]['duplicate_reason'] = f"Over-labeled: {p2['class']} (conf {p2['confidence']:.2f}) vs {p1['class']} (conf {p1['confidence']:.2f})"
                        overlabel_removed += 1
                    else:
                        particles[i]['deleted'] = True
                        particles[i]['duplicate_type'] = 'location_duplicate'
                        particles[i]['matched_with'] = j  # Link to kept particle
                        particles[i]['duplicate_reason'] = f"Over-labeled: {p1['class']} (conf {p1['confidence']:.2f}) vs {p2['class']} (conf {p2['confidence']:.2f})"
                        overlabel_removed += 1
                        break

    st.write(f"  ✅ Removed {overlabel_removed} over-labeled particles")

    final_active = [p for p in particles if not p.get('deleted')]
    final_deleted = [p for p in particles if p.get('deleted')]

    stats['after_overlabel_dedup'] = len(final_active)
    stats['final'] = len(final_active)

    # Summary
    st.divider()
    st.write("### 📊 Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Raw Detections", stats['raw'])
    col2.metric("Over-labeled Removed", overlabel_removed)
    col3.metric("Final Active", len(final_active))

    return particles, stats


# ============================================================================
# SCENARIO 2: TILED WITH METADATA (NEIGHBOR INFO)
# ============================================================================

def handle_tiled_with_neighbors(raw_particles, tile_metadata, tile_files, stats):
    """
    Tiled images with neighbor metadata:
    - IOU dedup for overlapping particles
    - Smart stitching for cut particles
    - Over-labeling detection
    """
    st.write("**Mode: Tiled with neighbor metadata (full dedup + stitching)**")

    particles = raw_particles.copy()

    # Parse coordinates
    tile_coords = {}
    for i, tile_meta in enumerate(tile_metadata):
        bbox = tile_meta.get('bbox_in_mosaic', [0, 0, 0, 0])
        tile_coords[i] = {
            'x_min': bbox[0],
            'y_min': bbox[1],
            'x_max': bbox[2],
            'y_max': bbox[3],
            'width': tile_meta.get('width', 0),
            'height': tile_meta.get('height', 0),
        }

    # Build neighbor graph from row/col
    neighbors = {}
    tile_grid = {}

    for i, tile_meta in enumerate(tile_metadata):
        row = tile_meta.get('row', i)
        col = tile_meta.get('col', 0)
        tile_grid[(row, col)] = i
        neighbors[i] = []

    for i, tile_meta in enumerate(tile_metadata):
        row = tile_meta.get('row', i)
        col = tile_meta.get('col', 0)

        for neighbor_row, neighbor_col in [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]:
            if (neighbor_row, neighbor_col) in tile_grid:
                neighbor_id = tile_grid[(neighbor_row, neighbor_col)]
                neighbors[i].append(neighbor_id)

    neighbor_pairs = sum(len(n) for n in neighbors.values()) // 2
    st.write(f"  Found {neighbor_pairs} neighboring tile pairs")

    # Convert to global coordinates and mark seams
    for p in particles:
        tile_id = p.get('tile_id', 0)
        if tile_id in tile_coords:
            t = tile_coords[tile_id]
            p['global_x'] = p.get('x', 0) + t['x_min']
            p['global_y'] = p.get('y', 0) + t['y_min']
        else:
            p['global_x'] = p.get('x', 0)
            p['global_y'] = p.get('y', 0)

        tile_w = tile_coords.get(tile_id, {}).get('width', 2880)
        tile_h = tile_coords.get(tile_id, {}).get('height', 2160)

        has_neighbors = tile_id in neighbors and len(neighbors[tile_id]) > 0

        if has_neighbors:
            extends_left = p.get('x', 0) < 5
            extends_right = (p.get('x', 0) + p.get('w', 0)) > (tile_w - 5)
            extends_top = p.get('y', 0) < 5
            extends_bottom = (p.get('y', 0) + p.get('h', 0)) > (tile_h - 5)

            p['at_seam'] = extends_left or extends_right or extends_top or extends_bottom
        else:
            p['at_seam'] = False

    particles.sort(key=lambda p: p.get('confidence', 0), reverse=True)

    # Step 1: IOU Dedup (skip particles extending to boundary)
    st.write("**Step 1: IOU Deduplication**")

    def iou_global(box1, box2):
        x1_min, y1_min = box1[0], box1[1]
        x1_max = x1_min + box1[2]
        y1_max = y1_min + box1[3]

        x2_min, y2_min = box2[0], box2[1]
        x2_max = x2_min + box2[2]
        y2_max = y2_min + box2[3]

        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)

        if xi_max < xi_min or yi_max < yi_min:
            return 0.0

        inter = (xi_max - xi_min) * (yi_max - yi_min)
        union = box1[2] * box1[3] + box2[2] * box2[3] - inter
        return inter / union if union > 0 else 0.0

    iou_dedup = []
    iou_removed = 0
    iou_threshold = 0.3

    # Keep track of original indices for matched_with field
    particle_indices = {}  # maps particle in iou_dedup to its original index in particles

    for orig_idx, particle in enumerate(particles):
        is_dup = False
        tile_id_1 = particle.get('tile_id', 0)

        # Skip particles extending to boundary (those are cuts)
        if particle.get('at_seam'):
            particle['deleted'] = False
            particle_indices[len(iou_dedup)] = orig_idx
            iou_dedup.append(particle)
            continue

        # Compare non-cut particles against neighbors
        if tile_id_1 in neighbors:
            box1 = (particle['global_x'], particle['global_y'], particle['w'], particle['h'])

            for dedup_idx, kept in enumerate(iou_dedup):
                if kept.get('at_seam'):
                    continue

                tile_id_2 = kept.get('tile_id', 0)
                if tile_id_2 not in neighbors[tile_id_1]:
                    continue

                size_match = abs(particle['diameter_um'] - kept['diameter_um']) / max(particle['diameter_um'], kept['diameter_um'], 0.1) < 0.2
                class_match = particle.get('class') == kept.get('class')

                if not (size_match and class_match):
                    continue

                box2 = (kept['global_x'], kept['global_y'], kept['w'], kept['h'])
                if iou_global(box1, box2) > iou_threshold:
                    is_dup = True
                    iou_removed += 1
                    particle['deleted'] = True
                    particle['duplicate_type'] = 'overlap_duplicate'
                    particle['duplicate_reason'] = f"IOU dedup: overlapping detection"
                    # Link to kept particle!
                    particle['matched_with'] = particle_indices[dedup_idx]
                    break

        if not is_dup:
            particle['deleted'] = False

        particle_indices[len(iou_dedup)] = orig_idx
        iou_dedup.append(particle)

    st.write(f"  ✅ Removed {iou_removed} overlapping detections")
    stats['after_iou_dedup'] = len([p for p in iou_dedup if not p.get('deleted')])

    # Step 2: Over-labeling
    st.write("**Step 2: Over-labeling detection**")

    overlabel_removed = 0
    particles_after = iou_dedup.copy()

    for i, p1 in enumerate(particles_after):
        if p1.get('deleted'):
            continue

        for j, p2 in enumerate(particles_after):
            if j <= i or p2.get('deleted'):
                continue

            if p1.get('tile_id') != p2.get('tile_id'):
                continue

            dist = ((p1['global_x'] - p2['global_x'])**2 + (p1['global_y'] - p2['global_y'])**2)**0.5

            if dist < 20 and p1.get('class') != p2.get('class'):
                if p1['confidence'] > p2['confidence']:
                    # p2 is lower confidence, delete it
                    particles_after[j]['deleted'] = True
                    particles_after[j]['duplicate_type'] = 'location_duplicate'
                    particles_after[j]['matched_with'] = i  # Link to kept particle
                    particles_after[j]['duplicate_reason'] = f"Over-labeled: {p2['class']} vs {p1['class']}"
                    overlabel_removed += 1
                else:
                    # p1 is lower confidence, delete it
                    particles_after[i]['deleted'] = True
                    particles_after[i]['duplicate_type'] = 'location_duplicate'
                    particles_after[i]['matched_with'] = j  # Link to kept particle
                    particles_after[i]['duplicate_reason'] = f"Over-labeled: {p1['class']} vs {p2['class']}"
                    overlabel_removed += 1
                    break

    st.write(f"  ✅ Removed {overlabel_removed} over-labeled particles")
    stats['after_overlabel_dedup'] = len([p for p in particles_after if not p.get('deleted')])

    # Step 3: Stitching
    st.write("**Step 3: Stitching cut particles**")

    at_seams = [p for p in particles_after if not p.get('deleted') and p.get('at_seam')]
    st.write(f"  Found {len(at_seams)} particles at seams")

    try:
        matcher_metadata = []
        for i, tm in enumerate(tile_metadata):
            matcher_metadata.append({
                "tile_id": i,
                "filename": tm.get('source_file') or tm.get('filename'),
                "width": tm.get('width', 2880),
                "height": tm.get('height', 2160),
                "neighbors": neighbors.get(i, [])
            })

        matcher = IntelligentParticleMatcher(matcher_metadata, size_tolerance=0.10, edge_margin_pct=0.05, confidence_threshold=0.5)
        stitched_particles, edge_matches = matcher.process_all_neighbors(particles_after)

        stitched_count = len([p for p in stitched_particles if p.get('stitched')])
        st.write(f"  ✅ Stitched {stitched_count} cut particles")
        stats['stitched_pairs'] = stitched_count
    except Exception as e:
        st.warning(f"Stitching error: {e}")
        stitched_particles = particles_after

    final_active = [p for p in stitched_particles if not p.get('deleted')]
    stats['final'] = len(final_active)

    # Summary
    st.divider()
    st.write("### 📊 Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Raw", stats['raw'])
    col2.metric("After IOU", stats['after_iou_dedup'])
    col3.metric("After Over-label", stats['after_overlabel_dedup'])
    col4.metric("Final", stats['final'])

    return stitched_particles, stats


# ============================================================================
# SCENARIO 3: STITCHED IMAGES (NO EDGE DEDUP, JUST CUTS + OVER-LABELING)
# ============================================================================

def handle_stitched_images(raw_particles, tile_metadata, tile_files, stats):
    """
    Already stitched images:
    - Skip edge duplicate detection (images already aligned)
    - Still handle cut particles
    - Still handle over-labeling
    """
    st.write("**Mode: Stitched images (skip edge dedup, handle cuts + over-labeling)**")

    particles = raw_particles.copy()

    # Over-labeling only (no need for global coords since stitched)
    st.write("**Step 1: Over-labeling detection**")

    overlabel_removed = 0

    for i, p1 in enumerate(particles):
        if p1.get('deleted'):
            continue

        for j, p2 in enumerate(particles):
            if j <= i or p2.get('deleted'):
                continue

            p1_x = p1.get('x', 0) + p1.get('w', 0) / 2
            p1_y = p1.get('y', 0) + p1.get('h', 0) / 2
            p2_x = p2.get('x', 0) + p2.get('w', 0) / 2
            p2_y = p2.get('y', 0) + p2.get('h', 0) / 2

            dist = ((p1_x - p2_x)**2 + (p1_y - p2_y)**2)**0.5

            if dist < 20 and p1.get('class') != p2.get('class'):
                if p1['confidence'] > p2['confidence']:
                    particles[j]['deleted'] = True
                    particles[j]['duplicate_type'] = 'location_duplicate'
                    particles[j]['matched_with'] = i  # Link to kept particle
                    overlabel_removed += 1
                else:
                    particles[i]['deleted'] = True
                    particles[i]['duplicate_type'] = 'location_duplicate'
                    particles[i]['matched_with'] = j  # Link to kept particle
                    overlabel_removed += 1
                    break

    st.write(f"  ✅ Removed {overlabel_removed} over-labeled particles")
    stats['after_overlabel_dedup'] = len([p for p in particles if not p.get('deleted')])

    final_active = [p for p in particles if not p.get('deleted')]
    stats['final'] = len(final_active)

    # Summary
    st.divider()
    st.write("### 📊 Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Raw Detections", stats['raw'])
    col2.metric("Over-labeled Removed", overlabel_removed)
    col3.metric("Final Active", len(final_active))

    return particles, stats