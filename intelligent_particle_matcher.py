"""
Intelligent edge-based particle matching and stitching
Matches particles across tile boundaries using multiple criteria
"""

import numpy as np
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
    
    def stitch_particles(self, match: Dict, tile_images: Dict, calibration_um_per_pixel: float = 1.299) -> Optional[Dict]:
        """
        Stitch two matched particles and recalculate size
        
        Args:
            match: Match record from find_matches_for_edge
            tile_images: Dict mapping tile_id to numpy array (BGR image)
            calibration_um_per_pixel: Pixel to micrometer calibration
        
        Returns:
            Stitched info dict with recalculated size, or None if failed
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
            
            # Calculate stitched dimensions and particle bounds
            if direction == "right":
                # Tiles side-by-side horizontally
                # tile1 is on left, tile2 is on right
                
                # In stitched image, tile2 starts at x = tile1_w
                x2_in_stitched = x2 + tile1_w
                
                # Find bounding box of both particles in stitched image
                x_min = min(x1, x2_in_stitched)
                x_max = max(x1 + w1, x2_in_stitched + w2)
                y_min = min(y1, y2)
                y_max = max(y1 + h1, y2 + h2)
                
                stitched_w = int(x_max - x_min)
                stitched_h = int(y_max - y_min)
                
                # Create stitched image for visualization (optional)
                stitched_img = np.concatenate([img1, img2], axis=1)
            
            elif direction == "bottom":
                # Tiles stacked vertically
                # tile1 is on top, tile2 is on bottom
                
                # In stitched image, tile2 starts at y = tile1_h
                y2_in_stitched = y2 + tile1_h
                
                # Find bounding box of both particles
                x_min = min(x1, x2)
                x_max = max(x1 + w1, x2 + w2)
                y_min = min(y1, y2_in_stitched)
                y_max = max(y1 + h1, y2_in_stitched + h2)
                
                stitched_w = int(x_max - x_min)
                stitched_h = int(y_max - y_min)
                
                # Create stitched image for visualization
                stitched_img = np.concatenate([img1, img2], axis=0)
            
            else:
                return None
            
            # Recalculate diameter on stitched image
            # Use diagonal of bounding box as diameter estimate
            stitched_diagonal_px = np.sqrt(stitched_w**2 + stitched_h**2)
            stitched_diameter_um = stitched_diagonal_px * calibration_um_per_pixel
            
            # Calculate size change
            original_diameter = p1.get("diameter_um", 0)
            size_change_pct = 0
            if original_diameter > 0:
                size_change_pct = ((stitched_diameter_um - original_diameter) / original_diameter) * 100
            
            return {
                "stitched": True,
                "original_tile1": tile1_id,
                "original_tile2": tile2_id,
                "direction": direction,
                "stitched_w_px": stitched_w,
                "stitched_h_px": stitched_h,
                "stitched_diagonal_px": stitched_diagonal_px,
                "stitched_diameter_um": stitched_diameter_um,
                "original_diameter_um": original_diameter,
                "size_change_pct": size_change_pct,
                "match_score": match.get("match_score", 0)
            }
        
        except Exception as e:
            print(f"Stitching error: {e}")
            return None