# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

"""
Occlusion and Overlap Handler for Enhanced StrongSort

This module provides comprehensive handling of occlusion scenarios to prevent
ID switching when people overlap or occlude each other.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum


class OcclusionType(Enum):
    """Types of occlusion scenarios"""
    NO_OCCLUSION = 0
    PARTIAL_OCCLUSION = 1
    FULL_OCCLUSION = 2
    MUTUAL_OCCLUSION = 3
    CROWD_OCCLUSION = 4


@dataclass
class OcclusionEvent:
    """Represents an occlusion event between tracks"""
    occluder_id: int
    occluded_id: int
    occlusion_type: OcclusionType
    start_frame: int
    end_frame: Optional[int] = None
    overlap_ratio: float = 0.0
    confidence: float = 0.0


class OverlapAnalyzer:
    """Analyzes overlapping relationships between bounding boxes"""
    
    def __init__(self, overlap_threshold: float = 0.3, crowd_threshold: int = 3):
        self.overlap_threshold = overlap_threshold
        self.crowd_threshold = crowd_threshold
        
    def compute_overlap_matrix(self, boxes: np.ndarray) -> np.ndarray:
        """Compute overlap matrix between all pairs of boxes"""
        if len(boxes) == 0:
            return np.array([])
        
        # Convert to [x1, y1, x2, y2] if needed
        if boxes.shape[1] == 4:
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        else:
            # Assume TLWH format: [x, y, w, h]
            x1, y1 = boxes[:, 0], boxes[:, 1]
            x2, y2 = boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]
        
        # Compute areas
        areas = (x2 - x1) * (y2 - y1)
        
        # Initialize overlap matrix
        n = len(boxes)
        overlap_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                # Intersection coordinates
                xx1 = max(x1[i], x1[j])
                yy1 = max(y1[i], y1[j])
                xx2 = min(x2[i], x2[j])
                yy2 = min(y2[i], y2[j])
                
                # Intersection area
                w = max(0, xx2 - xx1)
                h = max(0, yy2 - yy1)
                intersection = w * h
                
                if intersection > 0:
                    # Compute overlap ratios
                    overlap_i = intersection / areas[i]
                    overlap_j = intersection / areas[j]
                    
                    # Store maximum overlap ratio
                    overlap_matrix[i, j] = max(overlap_i, overlap_j)
                    overlap_matrix[j, i] = max(overlap_i, overlap_j)
        
        return overlap_matrix
    
    def detect_occlusion_type(self, overlap_ratio: float, relative_size: float) -> OcclusionType:
        """Determine occlusion type based on overlap and size ratios"""
        if overlap_ratio < self.overlap_threshold:
            return OcclusionType.NO_OCCLUSION
        elif overlap_ratio < 0.6:
            return OcclusionType.PARTIAL_OCCLUSION
        elif relative_size > 1.5:  # Larger object likely occluding smaller
            return OcclusionType.FULL_OCCLUSION
        else:
            return OcclusionType.MUTUAL_OCCLUSION
    
    def analyze_spatial_relationships(self, boxes: np.ndarray) -> Dict[str, np.ndarray]:
        """Analyze spatial relationships between boxes"""
        if len(boxes) == 0:
            return {}
        
        # Extract centers and dimensions
        if boxes.shape[1] == 4:  # XYXY format
            centers = np.column_stack([
                (boxes[:, 0] + boxes[:, 2]) / 2,
                (boxes[:, 1] + boxes[:, 3]) / 2
            ])
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
        else:  # TLWH format
            centers = np.column_stack([
                boxes[:, 0] + boxes[:, 2] / 2,
                boxes[:, 1] + boxes[:, 3] / 2
            ])
            widths = boxes[:, 2]
            heights = boxes[:, 3]
        
        areas = widths * heights
        
        # Compute distance matrix
        n = len(boxes)
        distance_matrix = np.zeros((n, n))
        size_ratio_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                # Euclidean distance between centers
                dist = np.linalg.norm(centers[i] - centers[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
                
                # Size ratio
                ratio = areas[i] / areas[j] if areas[j] > 0 else 1.0
                size_ratio_matrix[i, j] = ratio
                size_ratio_matrix[j, i] = 1.0 / ratio
        
        return {
            'distance_matrix': distance_matrix,
            'size_ratio_matrix': size_ratio_matrix,
            'centers': centers,
            'areas': areas
        }


class OcclusionStateManager:
    """Manages occlusion states and events for tracks"""
    
    def __init__(self, max_history: int = 50):
        self.max_history = max_history
        self.occlusion_events: Dict[int, List[OcclusionEvent]] = defaultdict(list)
        self.current_occlusions: Dict[int, Set[int]] = defaultdict(set)  # track_id -> set of occluding tracks
        self.occlusion_history: deque = deque(maxlen=max_history)
        self.track_visibility: Dict[int, float] = {}  # track_id -> visibility score
        
    def update_occlusion_state(self, track_boxes: Dict[int, np.ndarray], 
                              overlap_analyzer: OverlapAnalyzer, frame_id: int):
        """Update occlusion states for all tracks"""
        if not track_boxes:
            return
        
        track_ids = list(track_boxes.keys())
        boxes = np.array([track_boxes[tid] for tid in track_ids])
        
        # Compute overlap matrix
        overlap_matrix = overlap_analyzer.compute_overlap_matrix(boxes)
        spatial_info = overlap_analyzer.analyze_spatial_relationships(boxes)
        
        # Clear current occlusions
        self.current_occlusions.clear()
        
        # Analyze occlusions
        for i, track_i in enumerate(track_ids):
            visibility_score = 1.0
            
            for j, track_j in enumerate(track_ids):
                if i == j:
                    continue
                
                overlap_ratio = overlap_matrix[i, j]
                if overlap_ratio > overlap_analyzer.overlap_threshold:
                    size_ratio = spatial_info['size_ratio_matrix'][i, j]
                    occlusion_type = overlap_analyzer.detect_occlusion_type(overlap_ratio, size_ratio)
                    
                    if occlusion_type != OcclusionType.NO_OCCLUSION:
                        # Determine who is occluding whom
                        if size_ratio > 1.2:  # track_i is larger
                            occluder_id, occluded_id = track_i, track_j
                        elif size_ratio < 0.8:  # track_j is larger
                            occluder_id, occluded_id = track_j, track_i
                        else:  # Mutual occlusion
                            # Use depth cues or previous history
                            occluder_id, occluded_id = self._resolve_mutual_occlusion(
                                track_i, track_j, spatial_info, frame_id
                            )
                        
                        # Update current occlusions
                        self.current_occlusions[occluded_id].add(occluder_id)
                        
                        # Update visibility score
                        if track_i == occluded_id:
                            visibility_score *= (1.0 - overlap_ratio)
                        
                        # Record occlusion event
                        self._record_occlusion_event(
                            occluder_id, occluded_id, occlusion_type, 
                            overlap_ratio, frame_id
                        )
            
            self.track_visibility[track_i] = visibility_score
        
        # Store frame occlusion info
        self.occlusion_history.append({
            'frame_id': frame_id,
            'occlusions': dict(self.current_occlusions),
            'visibility': dict(self.track_visibility)
        })
    
    def _resolve_mutual_occlusion(self, track_i: int, track_j: int, 
                                 spatial_info: Dict, frame_id: int) -> Tuple[int, int]:
        """Resolve mutual occlusion using additional cues"""
        # Use motion direction and history to determine occlusion order
        # For now, use simple heuristic based on y-coordinate (lower object occludes higher)
        centers = spatial_info['centers']
        i_idx = list(spatial_info.get('track_indices', {track_i: 0, track_j: 1}).get(track_i, 0))
        j_idx = list(spatial_info.get('track_indices', {track_i: 0, track_j: 1}).get(track_j, 1))
        
        if len(centers) > max(i_idx, j_idx):
            if centers[i_idx][1] > centers[j_idx][1]:  # track_i is lower (closer to camera)
                return track_i, track_j
            else:
                return track_j, track_i
        
        # Fallback: use track ID (arbitrary but consistent)
        return (track_i, track_j) if track_i < track_j else (track_j, track_i)
    
    def _record_occlusion_event(self, occluder_id: int, occluded_id: int, 
                               occlusion_type: OcclusionType, overlap_ratio: float, frame_id: int):
        """Record an occlusion event"""
        # Check if this is a continuation of existing event
        existing_events = self.occlusion_events[occluded_id]
        if existing_events:
            last_event = existing_events[-1]
            if (last_event.occluder_id == occluder_id and 
                last_event.end_frame is None and 
                frame_id - last_event.start_frame < 10):  # Within 10 frames
                # Continue existing event
                last_event.overlap_ratio = max(last_event.overlap_ratio, overlap_ratio)
                return
        
        # Create new event
        event = OcclusionEvent(
            occluder_id=occluder_id,
            occluded_id=occluded_id,
            occlusion_type=occlusion_type,
            start_frame=frame_id,
            overlap_ratio=overlap_ratio,
            confidence=min(overlap_ratio * 2, 1.0)
        )
        self.occlusion_events[occluded_id].append(event)
    
    def is_track_occluded(self, track_id: int) -> bool:
        """Check if a track is currently occluded"""
        return len(self.current_occlusions.get(track_id, set())) > 0
    
    def get_occlusion_level(self, track_id: int) -> float:
        """Get occlusion level for a track (0=visible, 1=fully occluded)"""
        return 1.0 - self.track_visibility.get(track_id, 1.0)
    
    def get_occluding_tracks(self, track_id: int) -> Set[int]:
        """Get set of tracks that are occluding the given track"""
        return self.current_occlusions.get(track_id, set())
    
    def predict_track_position(self, track_id: int, occluder_boxes: Dict[int, np.ndarray]) -> Optional[np.ndarray]:
        """Predict position of occluded track based on occluders"""
        if not self.is_track_occluded(track_id):
            return None
        
        occluders = self.get_occluding_tracks(track_id)
        if not occluders or not occluder_boxes:
            return None
        
        # Simple prediction: use centroid of occluding boxes
        occluder_centers = []
        for occluder_id in occluders:
            if occluder_id in occluder_boxes:
                box = occluder_boxes[occluder_id]
                if len(box) >= 4:
                    if len(box) == 4:  # XYXY
                        center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
                    else:  # TLWH
                        center = [box[0] + box[2] / 2, box[1] + box[3] / 2]
                    occluder_centers.append(center)
        
        if occluder_centers:
            # Return approximate position under occluders
            mean_center = np.mean(occluder_centers, axis=0)
            # Estimate size based on recent history (simplified)
            estimated_size = [50, 100]  # Default person size
            return np.array([
                mean_center[0] - estimated_size[0] / 2,
                mean_center[1] - estimated_size[1] / 2,
                estimated_size[0],
                estimated_size[1]
            ])
        
        return None


class OcclusionAwareTracker:
    """Enhanced tracker with occlusion awareness"""
    
    def __init__(self, base_tracker, occlusion_threshold: float = 0.3):
        self.base_tracker = base_tracker
        self.overlap_analyzer = OverlapAnalyzer(overlap_threshold=occlusion_threshold)
        self.occlusion_manager = OcclusionStateManager()
        self.occlusion_buffer = {}  # Store features during occlusion
        
        # Enhanced parameters for occlusion handling
        self.occlusion_max_age_multiplier = 2.0  # Keep occluded tracks longer
        self.occlusion_confidence_boost = 0.1    # Boost confidence for emerging tracks
        self.appearance_memory_frames = 10       # Frames to remember appearance
        
    def update_with_occlusion_handling(self, detections, tracks, frame_id: int):
        """Update tracking with occlusion-aware processing"""
        
        # Extract track boxes for occlusion analysis
        track_boxes = {}
        for track in tracks:
            track_boxes[track.id] = track.to_tlwh()
        
        # Update occlusion states
        self.occlusion_manager.update_occlusion_state(
            track_boxes, self.overlap_analyzer, frame_id
        )
        
        # Apply occlusion-aware modifications
        self._apply_occlusion_modifications(tracks, detections, frame_id)
        
        return tracks
    
    def _apply_occlusion_modifications(self, tracks, detections, frame_id: int):
        """Apply occlusion-aware modifications to tracks and detections"""
        
        for track in tracks:
            occlusion_level = self.occlusion_manager.get_occlusion_level(track.id)
            
            if occlusion_level > 0.3:  # Track is occluded
                # Extend track lifetime
                if hasattr(track, '_max_age'):
                    track._max_age = int(track._max_age * self.occlusion_max_age_multiplier)
                
                # Store appearance features for later recovery
                if hasattr(track, 'features') and track.features:
                    if track.id not in self.occlusion_buffer:
                        self.occlusion_buffer[track.id] = deque(maxlen=self.appearance_memory_frames)
                    self.occlusion_buffer[track.id].extend(track.features[-1:])
                
                # Adjust confidence and quality scores
                if hasattr(track, 'quality_score'):
                    # Maintain quality during occlusion
                    track.quality_score = max(track.quality_score, 0.6)
                
                # Predict position if fully occluded
                if occlusion_level > 0.8:
                    track_boxes = {t.id: t.to_tlwh() for t in tracks if t.id != track.id}
                    predicted_pos = self.occlusion_manager.predict_track_position(
                        track.id, track_boxes
                    )
                    if predicted_pos is not None:
                        # Update track position with prediction
                        self._update_track_with_predicted_position(track, predicted_pos)
            
            elif track.id in self.occlusion_buffer:
                # Track is emerging from occlusion
                self._handle_emerging_track(track, frame_id)
    
    def _update_track_with_predicted_position(self, track, predicted_pos):
        """Update track with predicted position during occlusion"""
        # Convert predicted position to track format
        if hasattr(track, 'mean') and len(predicted_pos) >= 4:
            # Update Kalman filter state
            x, y, w, h = predicted_pos
            cx, cy = x + w/2, y + h/2
            aspect_ratio = w / h if h > 0 else 1.0
            
            # Update mean state [cx, cy, a, h, ...]
            track.mean[0] = cx
            track.mean[1] = cy
            track.mean[2] = aspect_ratio
            track.mean[3] = h
            
            # Increase uncertainty during occlusion
            if hasattr(track, 'covariance'):
                uncertainty_factor = 1.5
                track.covariance[:4, :4] *= uncertainty_factor
    
    def _handle_emerging_track(self, track, frame_id: int):
        """Handle track emerging from occlusion"""
        # Boost confidence for emerging tracks
        if hasattr(track, 'conf'):
            track.conf = min(track.conf + self.occlusion_confidence_boost, 1.0)
        
        # Use stored appearance features for better matching
        if track.id in self.occlusion_buffer and hasattr(track, 'features'):
            stored_features = list(self.occlusion_buffer[track.id])
            if stored_features and track.features:
                # Blend current and stored features
                current_feature = track.features[-1]
                best_stored = max(stored_features, key=lambda f: np.linalg.norm(f))
                
                # Weighted combination
                blended_feature = 0.7 * current_feature + 0.3 * best_stored
                blended_feature /= (np.linalg.norm(blended_feature) + 1e-8)
                track.features[-1] = blended_feature
        
        # Clear occlusion buffer after emergence
        if track.id in self.occlusion_buffer:
            del self.occlusion_buffer[track.id]
    
    def get_occlusion_statistics(self) -> Dict:
        """Get comprehensive occlusion statistics"""
        stats = {
            'currently_occluded_tracks': len([tid for tid, occluders in self.occlusion_manager.current_occlusions.items() if occluders]),
            'total_occlusion_events': sum(len(events) for events in self.occlusion_manager.occlusion_events.values()),
            'average_visibility': np.mean(list(self.occlusion_manager.track_visibility.values())) if self.occlusion_manager.track_visibility else 1.0,
            'tracks_in_occlusion_buffer': len(self.occlusion_buffer),
        }
        
        # Occlusion type distribution
        occlusion_types = defaultdict(int)
        for events in self.occlusion_manager.occlusion_events.values():
            for event in events:
                occlusion_types[event.occlusion_type.name] += 1
        stats['occlusion_type_distribution'] = dict(occlusion_types)
        
        return stats


def compute_crowd_density(boxes: np.ndarray, image_size: Tuple[int, int]) -> float:
    """Compute crowd density in the scene"""
    if len(boxes) == 0:
        return 0.0
    
    img_width, img_height = image_size
    img_area = img_width * img_height
    
    # Compute total person area
    if boxes.shape[1] == 4:  # XYXY format
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
    else:  # TLWH format
        widths = boxes[:, 2]
        heights = boxes[:, 3]
    
    total_person_area = np.sum(widths * heights)
    density = total_person_area / img_area
    
    return min(density, 1.0)  # Cap at 1.0


def detect_crowd_situations(tracks, density_threshold: float = 0.3) -> bool:
    """Detect if the current scene is a crowd situation"""
    if len(tracks) < 3:
        return False
    
    # Extract bounding boxes
    boxes = []
    for track in tracks:
        if hasattr(track, 'to_tlwh'):
            boxes.append(track.to_tlwh())
    
    if not boxes:
        return False
    
    boxes = np.array(boxes)
    
    # Compute overlap statistics
    overlap_analyzer = OverlapAnalyzer()
    overlap_matrix = overlap_analyzer.compute_overlap_matrix(boxes)
    
    # Count high-overlap pairs
    high_overlap_pairs = np.sum(overlap_matrix > 0.3) // 2  # Divide by 2 for symmetric matrix
    total_pairs = len(tracks) * (len(tracks) - 1) // 2
    
    overlap_ratio = high_overlap_pairs / max(total_pairs, 1)
    
    return overlap_ratio > density_threshold 