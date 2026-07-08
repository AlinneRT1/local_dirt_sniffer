"""
PROPER DUPLICATE & STITCH PIPELINE

Flow:
1. IOU Deduplication (TileParticleManager) - removes overlapping tile detections
2. Over-labeling detection - same spot, different class
3. Cut particle detection & stitching - particles at seams
4. Final counts and display
"""

def run_full_dedup_and_stitch_pipeline(raw_particles, tile_metadata, tile_files):
    """
    Complete pipeline: IOU dedup → Over-labeling → Stitching
    
    Returns:
        (final_particles, stats_dict)
    """
    import st
    from intelligent_particle_matcher import IntelligentParticleMatcher
    
    stats = {
        'raw': len(raw_particles),
        'after_iou_dedup': 0,
        'after_overlabel_dedup': 0,
        'stitched_pairs': 0,
        'final': 0,
    }
    
    # ============================================================================
    # STEP 1: IOU-BASED DEDUPLICATION USING GLOBAL COORDINATES
    # ============================================================================
    # USE bbox_in_mosaic to convert to global coordinates and find neighbors
    st.write("**Step 1: IOU Deduplication (using bbox_in_mosaic coordinates)**")
    
    particles = raw_particles.copy()
    iou_threshold = 0.3
    seam_margin = 30
    
    # Parse tile metadata from bbox_in_mosaic
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
    
    # Build neighbor graph from bbox overlaps
    neighbors = {}
    for i in range(len(tile_metadata)):
        neighbors[i] = []
        for j in range(len(tile_metadata)):
            if i == j:
                continue
            
            t1 = tile_coords[i]
            t2 = tile_coords[j]
            
            # Check if tiles are adjacent/overlapping
            # Horizontally adjacent?
            if (t1['x_max'] >= t2['x_min'] and t1['x_min'] <= t2['x_max']):
                # Vertically adjacent?
                if (t1['y_max'] >= t2['y_min'] and t1['y_min'] <= t2['y_max']):
                    neighbors[i].append(j)
    
    st.write(f"  Found tile relationships: {[(i, n) for i, ns in neighbors.items() if ns for n in ns]}")
    
    # Convert all particles to global coordinates
    for p in particles:
        tile_id = p.get('tile_id', 0)
        if tile_id in tile_coords:
            t = tile_coords[tile_id]
            # Convert tile-local to global/mosaic
            p['global_x'] = p.get('x', 0) + t['x_min']
            p['global_y'] = p.get('y', 0) + t['y_min']
        else:
            p['global_x'] = p.get('x', 0)
            p['global_y'] = p.get('y', 0)
        
        # Check if at seam
        tile_w = tile_coords.get(tile_id, {}).get('width', 2880)
        tile_h = tile_coords.get(tile_id, {}).get('height', 2160)
        
        at_seam_x = p.get('x', 0) < seam_margin or (p.get('x', 0) + p.get('w', 0)) > (tile_w - seam_margin)
        at_seam_y = p.get('y', 0) < seam_margin or (p.get('y', 0) + p.get('h', 0)) > (tile_h - seam_margin)
        
        p['at_seam'] = at_seam_x or at_seam_y
        p['seams'] = []
        if at_seam_x and p.get('x', 0) < seam_margin:
            p['seams'].append('left')
        if at_seam_x and (p.get('x', 0) + p.get('w', 0)) > (tile_w - seam_margin):
            p['seams'].append('right')
        if at_seam_y and p.get('y', 0) < seam_margin:
            p['seams'].append('top')
        if at_seam_y and (p.get('y', 0) + p.get('h', 0)) > (tile_h - seam_margin):
            p['seams'].append('bottom')
    
    # Sort by confidence
    particles.sort(key=lambda p: p.get('confidence', 0), reverse=True)
    
    # IOU on global coordinates
    def iou_global(box1, box2):
        """IOU between (x, y, w, h) in GLOBAL coordinates"""
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
    
    for particle in particles:
        is_dup = False
        tile_id_1 = particle.get('tile_id', 0)
        
        # Only compare against particles in neighboring tiles (smart!)
        if tile_id_1 in neighbors:
            box1 = (particle['global_x'], particle['global_y'], particle['w'], particle['h'])
            
            for kept in iou_dedup:
                tile_id_2 = kept.get('tile_id', 0)
                
                # Only compare with neighbors
                if tile_id_2 not in neighbors[tile_id_1]:
                    continue
                
                # Check size/class match first
                size_match = abs(particle['diameter_um'] - kept['diameter_um']) / max(particle['diameter_um'], kept['diameter_um'], 0.1) < 0.2
                class_match = particle.get('class') == kept.get('class')
                
                if not (size_match and class_match):
                    continue
                
                # Do IOU on global coordinates
                box2 = (kept['global_x'], kept['global_y'], kept['w'], kept['h'])
                if iou_global(box1, box2) > iou_threshold:
                    is_dup = True
                    iou_removed += 1
                    particle['deleted'] = True
                    particle['duplicate_type'] = 'overlap_duplicate'
                    particle['duplicate_reason'] = f"IOU dedup: overlapping detection from tile {kept['tile_id']} (conf {kept['confidence']:.2f})"
                    break
        
        if not is_dup:
            particle['deleted'] = False
        
        iou_dedup.append(particle)
    
    st.write(f"  ✅ Removed {iou_removed} overlapping detections (IOU > {iou_threshold})")
    st.write(f"  ↳ Remaining: {len([p for p in iou_dedup if not p.get('deleted')])}")
    
    stats['after_iou_dedup'] = len([p for p in iou_dedup if not p.get('deleted')])
    
    # ============================================================================
    # STEP 2: OVER-LABELING DETECTION (same spot, different classes)
    # ============================================================================
    st.write("**Step 2: Over-labeling detection (same spot, different class)**")
    
    # Build spatial index for same-location particles
    overlabel_removed = 0
    particles_after_overlabel = iou_dedup.copy()
    
    for i, p1 in enumerate(particles_after_overlabel):
        if p1.get('deleted'):
            continue
        
        for j, p2 in enumerate(particles_after_overlabel):
            if j <= i or p2.get('deleted'):
                continue
            
            # Check if same location
            dist = ((p1['mosaic_x'] - p2['mosaic_x'])**2 + (p1['mosaic_y'] - p2['mosaic_y'])**2)**0.5
            
            if dist < 20:  # Within 20px = same spot
                # Different class?
                if p1.get('class') != p2.get('class'):
                    # Keep higher confidence
                    if p1['confidence'] > p2['confidence']:
                        particles_after_overlabel[j]['deleted'] = True
                        particles_after_overlabel[j]['duplicate_type'] = 'location_duplicate'
                        particles_after_overlabel[j]['duplicate_reason'] = f"Over-labeled: {p2['class']} (conf {p2['confidence']:.2f}) vs {p1['class']} (conf {p1['confidence']:.2f})"
                        overlabel_removed += 1
                    else:
                        particles_after_overlabel[i]['deleted'] = True
                        particles_after_overlabel[i]['duplicate_type'] = 'location_duplicate'
                        particles_after_overlabel[i]['duplicate_reason'] = f"Over-labeled: {p1['class']} (conf {p1['confidence']:.2f}) vs {p2['class']} (conf {p2['confidence']:.2f})"
                        overlabel_removed += 1
                        break
    
    st.write(f"  ✅ Removed {overlabel_removed} over-labeled particles")
    st.write(f"  ↳ Remaining: {len([p for p in particles_after_overlabel if not p.get('deleted')])}")
    
    stats['after_overlabel_dedup'] = len([p for p in particles_after_overlabel if not p.get('deleted')])
    
    # ============================================================================
    # STEP 3: CUT PARTICLE DETECTION & STITCHING
    # ============================================================================
    st.write("**Step 3: Cut particle detection & stitching**")
    
    # Count particles at seams before stitching
    at_seams = [p for p in particles_after_overlabel if not p.get('deleted') and p.get('at_seam')]
    st.write(f"  Found {len(at_seams)} particles at tile seams (potential cuts)")
    
    # Use IntelligentParticleMatcher for stitching
    try:
        from intelligent_particle_matcher import IntelligentParticleMatcher
        
        matcher_metadata = []
        for i, tm in enumerate(tile_metadata):
            matcher_metadata.append({
                "tile_id": i,
                "filename": tm.get('source_file') or tm.get('filename'),
                "width": tm.get('width', 2880),
                "height": tm.get('height', 2160),
                "neighbors": tm.get('neighbors', {})
            })
        
        matcher = IntelligentParticleMatcher(
            matcher_metadata,
            size_tolerance=0.10,   # STRICT: 10%
            edge_margin_pct=0.05,  # STRICT: 5%
            confidence_threshold=0.5
        )
        
        # Process stitching
        stitched_particles, edge_matches = matcher.process_all_neighbors(particles_after_overlabel)
        
        stitched_count = len([p for p in stitched_particles if p.get('stitched')])
        st.write(f"  ✅ Stitched {stitched_count} cut particles")
        st.write(f"  ↳ Found {len(edge_matches)} edge matches")
        
        stats['stitched_pairs'] = stitched_count
        
    except Exception as e:
        st.warning(f"Stitching error: {e}")
        stitched_particles = particles_after_overlabel
    
    # ============================================================================
    # FINAL COUNTS
    # ============================================================================
    final_active = [p for p in stitched_particles if not p.get('deleted')]
    final_deleted = [p for p in stitched_particles if p.get('deleted')]
    
    stats['final'] = len(final_active)
    
    st.divider()
    st.write("### 📊 Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Raw Detections", stats['raw'])
    col2.metric("After IOU Dedup", stats['after_iou_dedup'])
    col3.metric("After Over-label", stats['after_overlabel_dedup'])
    col4.metric("Final (Active)", stats['final'])
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("IOU Removed", stats['raw'] - stats['after_iou_dedup'])
    col2.metric("Over-label Removed", stats['after_iou_dedup'] - stats['after_overlabel_dedup'])
    col3.metric("Stitched Pairs", stats['stitched_pairs'])
    col4.metric("Total Deleted", len(final_deleted))
    
    return stitched_particles, stats
