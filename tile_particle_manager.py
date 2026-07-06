"""
Tile Deduplication & Cut Particle Detection
Handles particle detection across multiple tiles with overlap management
"""

import numpy as np
import json
from typing import List, Dict, Tuple


class TileParticleManager:
    """Manage particles across tiles with deduplication and cut detection"""
    
    def __init__(self, metadata_file: str, iou_threshold: float = 0.3, seam_margin: int = 30):
        """
        Initialize manager
        
        Args:
            metadata_file: JSON file with tile metadata
            iou_threshold: IOU threshold for deduplication (0-1)
            seam_margin: pixels from edge to consider as "at seam"
        """
        self.metadata = self._load_metadata(metadata_file)
        self.iou_threshold = iou_threshold
        self.seam_margin = seam_margin
        self.tiles_dict = {tile["id"]: tile for tile in self.metadata}
    
    def _load_metadata(self, metadata_file: str) -> List[Dict]:
        """Load tile metadata from JSON"""
        with open(metadata_file, 'r') as f:
            return json.load(f)
    
    def convert_to_mosaic_coords(self, tile_id: int, x: int, y: int) -> Tuple[int, int]:
        """Convert tile-local coordinates to mosaic coordinates"""
        tile = self.tiles_dict[tile_id]
        mosaic_x = x + tile["x_start"]
        mosaic_y = y + tile["y_start"]
        return mosaic_x, mosaic_y
    
    def is_at_seam(self, tile_id: int, x: int, y: int, w: int, h: int) -> Dict:
        """
        Check if particle is near tile seams (potentially cut)
        
        Returns dict with seam info:
        {
            "at_seam": bool,
            "seams": ["left", "right", "top", "bottom"]  # which seams nearby
        }
        """
        tile = self.tiles_dict[tile_id]
        tile_w = tile["x_end"] - tile["x_start"]
        tile_h = tile["y_end"] - tile["y_start"]
        
        seams = []
        
        # Check distance from edges
        if x < self.seam_margin:
            seams.append("left")
        if x + w > tile_w - self.seam_margin:
            seams.append("right")
        if y < self.seam_margin:
            seams.append("top")
        if y + h > tile_h - self.seam_margin:
            seams.append("bottom")
        
        return {
            "at_seam": len(seams) > 0,
            "seams": seams
        }
    
    def iou(self, box1: Tuple, box2: Tuple) -> float:
        """Calculate IOU between two boxes (x, y, w, h)"""
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
        
        inter_area = (xi_max - xi_min) * (yi_max - yi_min)
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def process_tile_particles(self, tile_particles: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        Process particles from all tiles:
        1. Convert to mosaic coordinates
        2. Deduplicate overlapping detections
        3. Mark particles at seams as potentially cut
        
        Args:
            tile_particles: List of particles with format:
            {
                "tile_id": int,
                "x": int (tile-local),
                "y": int (tile-local),
                "w": int,
                "h": int,
                "class": str,
                "confidence": float,
                "diameter_um": float,
                ...
            }
        
        Returns:
            (deduplicated_particles, dedup_stats)
        """
        
        # Step 1: Convert all to mosaic coordinates
        mosaic_particles = []
        for p in tile_particles:
            tile_id = p["tile_id"]
            mosaic_x, mosaic_y = self.convert_to_mosaic_coords(tile_id, p["x"], p["y"])
            seam_info = self.is_at_seam(tile_id, p["x"], p["y"], p["w"], p["h"])
            
            particle = p.copy()
            particle["mosaic_x"] = mosaic_x
            particle["mosaic_y"] = mosaic_y
            particle["at_seam"] = seam_info["at_seam"]
            particle["seams"] = seam_info["seams"]
            mosaic_particles.append(particle)
        
        # Step 2: Deduplicate (keep highest confidence)
        mosaic_particles.sort(key=lambda p: p["confidence"], reverse=True)
        deduplicated = []
        duplicates_removed = 0
        
        for particle in mosaic_particles:
            is_duplicate = False
            box1 = (particle["mosaic_x"], particle["mosaic_y"], particle["w"], particle["h"])
            
            for kept in deduplicated:
                box2 = (kept["mosaic_x"], kept["mosaic_y"], kept["w"], kept["h"])
                if self.iou(box1, box2) > self.iou_threshold:
                    is_duplicate = True
                    duplicates_removed += 1
                    break
            
            if not is_duplicate:
                deduplicated.append(particle)
        
        # Step 3: Statistics
        stats = {
            "total_input": len(tile_particles),
            "after_dedup": len(deduplicated),
            "duplicates_removed": duplicates_removed,
            "at_seams": len([p for p in deduplicated if p["at_seam"]]),
        }
        
        return deduplicated, stats
    
    def find_particles_at_neighbor_seams(self, particle: Dict, all_particles: List[Dict]) -> List[Dict]:
        """
        Find particles in neighboring tiles that might be part of the same particle cut across seam
        
        Returns list of particles in neighboring tiles at the shared seam boundary
        """
        tile_id = particle["tile_id"]
        tile = self.tiles_dict[tile_id]
        neighbors = tile.get("neighbors", {})
        
        potential_matches = []
        
        # Check each seam direction
        for seam_dir, neighbor_filename in neighbors.items():
            if neighbor_filename is None or seam_dir not in particle["seams"]:
                continue
            
            # Find neighbor tile
            neighbor_tile = None
            for t in self.metadata:
                if t["filename"] == neighbor_filename:
                    neighbor_tile = t
                    break
            
            if neighbor_tile is None:
                continue
            
            # Find particles in neighbor tile at the shared seam
            for other in all_particles:
                if other["tile_id"] != neighbor_tile["id"]:
                    continue
                
                # Check if at shared seam boundary
                if seam_dir == "left" and other["x"] + other["w"] > (neighbor_tile["x_end"] - neighbor_tile["x_start"] - self.seam_margin):
                    potential_matches.append(other)
                elif seam_dir == "right" and other["x"] < self.seam_margin:
                    potential_matches.append(other)
                elif seam_dir == "top" and other["y"] + other["h"] > (neighbor_tile["y_end"] - neighbor_tile["y_start"] - self.seam_margin):
                    potential_matches.append(other)
                elif seam_dir == "bottom" and other["y"] < self.seam_margin:
                    potential_matches.append(other)
        
        return potential_matches
    
    def merge_cut_particles(self, deduplicated: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Attempt to merge particles that appear to be cut across seams
        
        Returns:
            (merged_particles, merged_pairs)
        """
        merged_pairs = []
        merged_particles = []
        merged_indices = set()
        
        for i, particle in enumerate(deduplicated):
            if i in merged_indices:
                continue
            
            if not particle["at_seam"]:
                merged_particles.append(particle)
                continue
            
            # Find potential matches in neighboring tiles
            matches = self.find_particles_at_neighbor_seams(particle, deduplicated)
            
            if matches:
                # Merge with closest match
                best_match = min(matches, key=lambda m: self.iou(
                    (particle["mosaic_x"], particle["mosaic_y"], particle["w"], particle["h"]),
                    (self.convert_to_mosaic_coords(m["tile_id"], m["x"], m["y"])[0], 
                     self.convert_to_mosaic_coords(m["tile_id"], m["x"], m["y"])[1], 
                     m["w"], m["h"])
                ))
                
                # Create merged particle
                merged = {
                    **particle,
                    "merged": True,
                    "merged_with": best_match["tile_id"],
                    "original_particles": [particle, best_match]
                }
                merged_particles.append(merged)
                merged_pairs.append((particle["tile_id"], best_match["tile_id"]))
                merged_indices.add(deduplicated.index(best_match))
            else:
                merged_particles.append(particle)
        
        return merged_particles, merged_pairs


# Example Usage
if __name__ == "__main__":
    # Initialize manager
    manager = TileParticleManager(
        metadata_file="tile_metadata.json",
        iou_threshold=0.3,
        seam_margin=30
    )
    
    # Example tile particles (from YOLO detection on each tile)
    tile_particles = [
        {
            "tile_id": 0,
            "x": 100, "y": 150,
            "w": 50, "h": 60,
            "class": "Fiber",
            "confidence": 0.95,
            "diameter_um": 65.0
        },
        {
            "tile_id": 1,
            "x": 10, "y": 200,
            "w": 50, "h": 60,
            "class": "Fiber",
            "confidence": 0.92,
            "diameter_um": 64.5
        }
    ]
    
    # Process
    deduplicated, stats = manager.process_tile_particles(tile_particles)
    print(f"Stats: {stats}")
    print(f"Deduplicated particles: {len(deduplicated)}")
    
    for p in deduplicated:
        print(f"  Particle: mosaic({p['mosaic_x']}, {p['mosaic_y']}) "
              f"at_seam={p['at_seam']} seams={p['seams']}")
