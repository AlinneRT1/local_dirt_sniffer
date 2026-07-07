"""
Intelligent edge-based particle matching and stitching
Matches particles across tile boundaries using multiple criteria
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from PIL import Image
from scipy import ndimage


class IntelligentParticleMatcher:
    """Match and stitch particles across tile boundaries intelligently"""

    def __init__(self, tile_metadata: List[Dict], size_tolerance: float = 0.20,
                 edge_margin_pct: float = 0.10, confidence_threshold: float = 0.5):
        """
        Args:
            tile_metadata: List of tile metadata dicts with neighbors
            size_tolerance: Particle size difference tolerance (0.20 = 20%)
            edge_margin_pct: Margin from edge to check for candidates (0.10 = 10%)
            confidence_threshold: Min confidence for a match to be accepted
        """
        self.tiles = {tm["tile_id"]: tm for tm in tile_metadata}
        self.tile_metadata = tile_metadata
        self.size_tolerance = size_tolerance
        self.edge_margin_pct = edge_margin_pct
        self.confidence_threshold = confidence_threshold

    def find_edge_particles(self, particles: List[Dict], tile_id: int,
                            direction: str) -> List[Tuple[int, Dict]]:
        """
        Find particles near an edge of a tile

        Args:
            particles: All particles
            tile_id: Which tile
            direction: "left", "right", "top", "bottom"

        Returns:
            List of (particle_idx, particle) tuples near the edge
        """
        tile = self.tiles[tile_id]
        tile_w = tile["width"]
        tile_h = tile["height"]
        margin = max(tile_w, tile_h) * self.edge_margin_pct

        edge_particles = []

        for idx, p in enumerate(particles):
            if p.get("tile_id") != tile_id or p.get("deleted"):
                continue

            x = p.get("x", 0)
            y = p.get("y", 0)
            w = p.get("w", 0)
            h = p.get("h", 0)

            # Check which edge
            if direction == "left" and x < margin:
                edge_particles.append((idx, p))
            elif direction == "right" and x + w > tile_w - margin:
                edge_particles.append((idx, p))
            elif direction == "top" and y < margin:
                edge_particles.append((idx, p))
            elif direction == "bottom" and y + h > tile_h - margin:
                edge_particles.append((idx, p))

        return edge_particles

    def particles_match(self, p1: Dict, p2: Dict, direction: str) -> Tuple[bool, Dict]:
        """
        Check if two particles match across an edge

        Returns:
            (match_bool, match_info_dict)
        """
        match_info = {
            "size_match": False,
            "class_match": False,
            "position_reasonable": False,
            "confidence_ok": False,
            "overall_score": 0.0
        }

        # Criterion 1: Similar size (within tolerance)
        d1 = p1.get("diameter_um", 0)
        d2 = p2.get("diameter_um", 0)
        if d1 > 0 and d2 > 0:
            size_diff = abs(d1 - d2) / max(d1, d2)
            if size_diff <= self.size_tolerance:
                match_info["size_match"] = True
                match_info["overall_score"] += 0.4

        # Criterion 2: Same class
        if p1.get("class") == p2.get("class"):
            match_info["class_match"] = True
            match_info["overall_score"] += 0.3

        # Criterion 3: Confidence high enough on at least one
        conf1 = p1.get("confidence", 0)
        conf2 = p2.get("confidence", 0)
        if conf1 >= self.confidence_threshold or conf2 >= self.confidence_threshold:
            match_info["confidence_ok"] = True
            match_info["overall_score"] += 0.2

        # Criterion 4: Position makes geometric sense
        # For particles at edges, they should be roughly aligned on the perpendicular axis
        y1 = p1.get("y", 0) + p1.get("h", 0) / 2
        y2 = p2.get("y", 0) + p2.get("h", 0) / 2
        x1 = p1.get("x", 0) + p1.get("w", 0) / 2
        x2 = p2.get("x", 0) + p2.get("w", 0) / 2

        max_h = max(p1.get("h", 0), p2.get("h", 0))
        max_w = max(p1.get("w", 0), p2.get("w", 0))

        # For left/right edges, Y should be similar
        if direction in ["left", "right"]:
            y_diff = abs(y1 - y2)
            if y_diff <= max_h * 0.5:  # Within half of max height
                match_info["position_reasonable"] = True
                match_info["overall_score"] += 0.1
        # For top/bottom edges, X should be similar
        elif direction in ["top", "bottom"]:
            x_diff = abs(x1 - x2)
            if x_diff <= max_w * 0.5:  # Within half of max width
                match_info["position_reasonable"] = True
                match_info["overall_score"] += 0.1

        # Overall match: need at least 3 criteria met and score > 0.5
        criteria_met = sum([
            match_info["size_match"],
            match_info["class_match"],
            match_info["confidence_ok"],
            match_info["position_reasonable"]
        ])

        is_match = criteria_met >= 3 and match_info["overall_score"] >= 0.7

        return is_match, match_info

    def find_matches_for_edge(self, particles: List[Dict], tile_id_1: int,
                              tile_id_2: int, direction: str) -> List[Dict]:
        """
        Find matching particles between two neighboring tiles

        Returns:
            List of match records: {
                "tile1_idx": int,
                "tile2_idx": int,
                "tile1_id": int,
                "tile2_id": int,
                "particle1": Dict,
                "particle2": Dict,
                "match_score": float,
                "direction": str
            }
        """
        # Direction from tile1's perspective
        edge1_particles = self.find_edge_particles(particles, tile_id_1, direction)

        # Direction from tile2's perspective (opposite)
        opposite_dir = {
            "left": "right",
            "right": "left",
            "top": "bottom",
            "bottom": "top"
        }[direction]
        edge2_particles = self.find_edge_particles(particles, tile_id_2, opposite_dir)

        matches = []
        matched_idx2 = set()

        for idx1, p1 in edge1_particles:
            best_match = None
            best_score = 0
            best_idx2 = None

            for idx2, p2 in edge2_particles:
                if idx2 in matched_idx2:
                    continue

                is_match, match_info = self.particles_match(p1, p2, direction)

                if is_match and match_info["overall_score"] > best_score:
                    best_match = match_info
                    best_score = match_info["overall_score"]
                    best_idx2 = idx2

            if best_match:
                matched_idx2.add(best_idx2)
                matches.append({
                    "tile1_idx": idx1,
                    "tile2_idx": best_idx2,
                    "tile1_id": tile_id_1,
                    "tile2_id": tile_id_2,
                    "particle1": edge1_particles[[i for i, _ in edge1_particles].index(idx1)][1],
                    "particle2": edge2_particles[[i for i, _ in edge2_particles].index(best_idx2)][1],
                    "match_score": best_score,
                    "direction": direction,
                    "match_info": best_match
                })

        return matches

    def process_all_neighbors(self, particles: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Find all matches across all tile neighbors

        Returns:
            (particles_with_matches, list_of_matches)
        """
        matches = []

        # Find all neighbor pairs
        for tile_meta in self.tile_metadata:
            tile_id = tile_meta["tile_id"]
            neighbors = tile_meta.get("neighbors", {})

            for direction, neighbor_filename in neighbors.items():
                if neighbor_filename is None:
                    continue

                # Find neighbor tile_id
                neighbor_id = None
                for tm in self.tile_metadata:
                    if tm["filename"] == neighbor_filename:
                        neighbor_id = tm["tile_id"]
                        break

                if neighbor_id is None:
                    continue

                # Only process each pair once (avoid duplicates)
                if tile_id < neighbor_id:
                    edge_matches = self.find_matches_for_edge(
                        particles, tile_id, neighbor_id, direction
                    )
                    matches.extend(edge_matches)

        # Mark particles with matches
        for match in matches:
            p1 = match["particle1"]
            p2 = match["particle2"]

            # Keep higher confidence, mark other for removal
            if p1.get("confidence", 0) >= p2.get("confidence", 0):
                p2["deleted"] = True
                p2["matched_with"] = match["tile1_idx"]
                p1["has_match"] = True
            else:
                p1["deleted"] = True
                p1["matched_with"] = match["tile2_idx"]
                p2["has_match"] = True

        return particles, matches

    def stitch_particles(self, match: Dict, tile_images: Dict, calibration_um_per_pixel: float = 1.299) -> Optional[
        Dict]:
        """
        Stitch two matched particles and recalculate size
        Creates stitched image with overlap based on particle positions
        Draws red boundary line at tile junction

        Args:
            match: Match record from find_matches_for_edge
            tile_images: Dict mapping tile_id to numpy array (RGB image)
            calibration_um_per_pixel: Pixel to micrometer calibration

        Returns:
            Stitched info dict with recalculated size and stitched image, or None if failed
        """
        try:
            p1 = match["particle1"]
            p2 = match["particle2"]
            tile1_id = match["tile1_id"]
            tile2_id = match["tile2_id"]
            direction = match["direction"]

            # Get tile images
            img1 = tile_images.get(tile1_id)
            img2 = tile_images.get(tile2_id)
            if img1 is None or img2 is None:
                return None

            # Ensure numpy arrays
            if not isinstance(img1, np.ndarray):
                img1 = np.array(img1)
            if not isinstance(img2, np.ndarray):
                img2 = np.array(img2)

            # Get particle coordinates
            x1 = p1.get("x", 0)
            y1 = p1.get("y", 0)
            w1 = p1.get("w", 0)
            h1 = p1.get("h", 0)

            x2 = p2.get("x", 0)
            y2 = p2.get("y", 0)
            w2 = p2.get("w", 0)
            h2 = p2.get("h", 0)

            tile1_h, tile1_w = img1.shape[:2]
            tile2_h, tile2_w = img2.shape[:2]

            # Calculate overlap distance based on particle cut positions
            overlap_pixels = 0
            boundary_line_pos = 0
            stitched_img = None

            if direction == "right":
                # Tiles side-by-side horizontally
                # Calculate how much particle 1 extends past tile edge
                p1_extends_past_edge = max(0, (x1 + w1) - tile1_w)

                # Intelligent overlap based on particle characteristics:
                # - Particle extension tells us how much is cut
                # - Particle width tells us size of overlap zone needed
                # - At least 1.5x the particle width for reliable stitching

                particle_width = w1
                min_overlap_for_particle = int(particle_width * 1.5)

                # Overlap should be: extension + additional safety margin
                overlap_pixels = max(
                    p1_extends_past_edge + int(particle_width * 0.5),  # Extension + 50% extra
                    min_overlap_for_particle  # At least 1.5x particle width
                )

                # Cap at 25% of tile width
                max_overlap = min(tile1_w, tile2_w) // 4
                overlap_pixels = min(overlap_pixels, max_overlap)
                overlap_pixels = max(50, overlap_pixels)  # At least 50px

                # Position of boundary line in stitched image
                boundary_line_pos = tile1_w - overlap_pixels

                # Create stitched image with overlap
                stitched_img = np.concatenate([img1[:, :-overlap_pixels], img2[:, overlap_pixels:]], axis=1)

                # Calculate particles in stitched space
                x2_in_stitched = x2 + (tile1_w - overlap_pixels)
                x_min = min(x1, x2_in_stitched)
                x_max = max(x1 + w1, x2_in_stitched + w2)
                y_min = min(y1, y2)
                y_max = max(y1 + h1, y2 + h2)
                stitched_w = int(x_max - x_min)
                stitched_h = int(y_max - y_min)

            elif direction == "bottom":
                # Tiles stacked vertically
                p1_extends_past_edge = max(0, (y1 + h1) - tile1_h)

                particle_height = h1
                min_overlap_for_particle = int(particle_height * 1.5)

                overlap_pixels = max(
                    p1_extends_past_edge + int(particle_height * 0.5),
                    min_overlap_for_particle
                )

                max_overlap = min(tile1_h, tile2_h) // 4
                overlap_pixels = min(overlap_pixels, max_overlap)
                overlap_pixels = max(50, overlap_pixels)

                boundary_line_pos = tile1_h - overlap_pixels

                stitched_img = np.concatenate([img1[:-overlap_pixels, :], img2[overlap_pixels:, :]], axis=0)

                y2_in_stitched = y2 + (tile1_h - overlap_pixels)
                x_min = min(x1, x2)
                x_max = max(x1 + w1, x2 + w2)
                y_min = min(y1, y2_in_stitched)
                y_max = max(y1 + h1, y2_in_stitched + h2)
                stitched_w = int(x_max - x_min)
                stitched_h = int(y_max - y_min)

            else:
                return None

            # Draw RED BOUNDARY LINE and BLUE BOUNDING BOX on stitched image
            stitched_img_marked = stitched_img.copy()
            if stitched_img_marked.dtype != np.uint8:
                stitched_img_marked = stitched_img_marked.astype(np.uint8)

            line_thickness = 4
            line_color = (255, 0, 0)  # Red in RGB

            try:
                if direction == "right":
                    # Vertical red line at boundary
                    cv2.line(stitched_img_marked,
                             (int(boundary_line_pos), 0),
                             (int(boundary_line_pos), stitched_img_marked.shape[0]),
                             line_color, line_thickness)
                elif direction == "bottom":
                    # Horizontal red line at boundary
                    cv2.line(stitched_img_marked,
                             (0, int(boundary_line_pos)),
                             (stitched_img_marked.shape[1], int(boundary_line_pos)),
                             line_color, line_thickness)

                # Draw BLUE BOUNDING BOX around complete particle ✅
                box_color = (0, 0, 255)  # Blue in RGB
                box_thickness = 3
                pt1 = (int(x_min), int(y_min))
                pt2 = (int(x_max), int(y_max))
                cv2.rectangle(stitched_img_marked, pt1, pt2, box_color, box_thickness)

            except:
                pass  # If cv2 fails, just use unmarked image

            # Recalculate diameter on stitched image
            # For cut particles, the stitched bbox should be larger than either partial bbox
            stitched_diagonal_px = np.sqrt(stitched_w ** 2 + stitched_h ** 2)
            stitched_diameter_um = stitched_diagonal_px * calibration_um_per_pixel

            # Calculate size change
            original_diameter = p1.get("diameter_um", 0)
            size_change_pct = 0

            # DEBUG: Check if particles actually got merged
            # If stitched size equals original, particles might not be overlapping correctly
            if original_diameter > 0:
                size_change_pct = ((stitched_diameter_um - original_diameter) / original_diameter) * 100

                # If size change is 0% or negative, particles aren't actually merging
                if size_change_pct <= 0:
                    print(
                        f"⚠️ WARNING: Stitched size not larger! Original: {original_diameter:.1f}µm, Stitched: {stitched_diameter_um:.1f}µm")
                    print(f"   P1: ({x1}, {y1}, {w1}×{h1}) on {tile1_w}×{tile1_h}")
                    print(f"   P2: ({x2}, {y2}, {w2}×{h2}) on {tile2_w}×{tile2_h}")
                    print(f"   Stitched bbox: {stitched_w}×{stitched_h}")
                    print(f"   Direction: {direction}, Overlap: {overlap_pixels}px")

            return {
                "stitched": True,
                "original_tile1": tile1_id,
                "original_tile2": tile2_id,
                "direction": direction,
                "overlap_pixels": overlap_pixels,
                "boundary_line_pos": boundary_line_pos,
                "stitched_image": stitched_img_marked,  # Image with red boundary line AND blue box ✅
                "stitched_w_px": stitched_w,
                "stitched_h_px": stitched_h,
                "stitched_diagonal_px": stitched_diagonal_px,
                "stitched_diameter_um": stitched_diameter_um,
                "original_diameter_um": original_diameter,
                "size_change_pct": size_change_pct,
                "match_score": match.get("match_score", 0),
                # Bounding box coordinates on stitched image ✅
                "bbox_x_min": int(x_min),
                "bbox_y_min": int(y_min),
                "bbox_x_max": int(x_max),
                "bbox_y_max": int(y_max)
            }

        except Exception as e:
            print(f"Stitching error: {e}")
            return None