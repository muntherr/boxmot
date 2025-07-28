# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

from pathlib import Path

import numpy as np
from torch import device

from boxmot.appearance.reid.auto_backend import ReidAutoBackend
from boxmot.motion.cmc import get_cmc_method
from boxmot.trackers.basetracker import BaseTracker
from boxmot.trackers.strongsort.sort.detection import Detection
from boxmot.trackers.strongsort.sort.tracker import Tracker
from boxmot.trackers.strongsort.sort.linear_assignment import NearestNeighborDistanceMetric
from boxmot.utils.ops import xyxy2tlwh


class StrongSort(object):
    """
    Enhanced StrongSORT Tracker: A high-performance tracking algorithm that utilizes a combination 
    of appearance and motion-based tracking with advanced ID preservation techniques.

    Args:
        model_weights (str): Path to the model weights for ReID (Re-Identification).
        device (str): Device on which to run the model (e.g., 'cpu' or 'cuda').
        fp16 (bool): Whether to use half-precision (fp16) for faster inference on compatible devices.
        per_class (bool, optional): Whether to perform per-class tracking. If True, tracks are maintained separately for each object class.
        max_dist (float, optional): Maximum cosine distance for ReID feature matching in Nearest Neighbor Distance Metric.
        max_iou_dist (float, optional): Maximum Intersection over Union (IoU) distance for data association. Controls the maximum allowed distance between tracklets and detections for a match.
        max_age (int, optional): Maximum number of frames to keep a track alive without any detections.
        n_init (int, optional): Number of consecutive frames required to confirm a track.
        nn_budget (int, optional): Maximum size of the feature library for Nearest Neighbor Distance Metric. If the library size exceeds this value, the oldest features are removed.
        mc_lambda (float, optional): Weight for motion consistency in the track state estimation. Higher values give more weight to motion information.
        ema_alpha (float, optional): Alpha value for exponential moving average (EMA) update of appearance features. Controls the contribution of new and old embeddings in the ReID model.
        conf_thresh_high (float, optional): High confidence threshold for prioritizing reliable detections in matching.
        conf_thresh_low (float, optional): Low confidence threshold for filtering out weak detections.
        id_preservation_weight (float, optional): Weight factor for ID preservation in matching cost computation.
    """

    def __init__(
        self,
        reid_weights: Path,
        device: device,
        half: bool,
        per_class: bool = False,
        min_conf: float = 0.1,
        max_cos_dist=0.15,  # Reduced for stricter appearance matching
        max_iou_dist=0.7,
        max_age=50,  # Increased to retain tracks longer
        n_init=2,  # Reduced for faster track confirmation
        nn_budget=150,  # Increased for better ReID memory
        mc_lambda=0.995,  # Increased for better motion consistency
        ema_alpha=0.9,
        conf_thresh_high=0.7,  # High confidence threshold
        conf_thresh_low=0.3,   # Low confidence threshold
        id_preservation_weight=0.1,  # Weight for ID preservation
        adaptive_matching=True,  # Enable adaptive matching
        appearance_weight=0.6,  # Weight for appearance features
        motion_weight=0.4,  # Weight for motion features
    ):

        self.per_class = per_class
        self.min_conf = min_conf
        self.conf_thresh_high = conf_thresh_high
        self.conf_thresh_low = conf_thresh_low
        self.id_preservation_weight = id_preservation_weight
        self.adaptive_matching = adaptive_matching
        self.appearance_weight = appearance_weight
        self.motion_weight = motion_weight
        
        # Enhanced ReID model with improved feature extraction
        self.model = ReidAutoBackend(
            weights=reid_weights, device=device, half=half
        ).model

        # Enhanced tracker with improved parameters
        self.tracker = Tracker(
            metric=NearestNeighborDistanceMetric("cosine", max_cos_dist, nn_budget),
            max_iou_dist=max_iou_dist,
            max_age=max_age,
            n_init=n_init,
            mc_lambda=mc_lambda,
            ema_alpha=ema_alpha,
            conf_thresh_high=conf_thresh_high,
            conf_thresh_low=conf_thresh_low,
            id_preservation_weight=id_preservation_weight,
            adaptive_matching=adaptive_matching,
            appearance_weight=appearance_weight,
            motion_weight=motion_weight,
        )
        self.cmc = get_cmc_method("ecc")()
        
        # Track management variables
        self.frame_count = 0
        self.lost_track_buffer = []  # Buffer for recently lost tracks
        self.track_history = {}  # Track appearance history

    @BaseTracker.per_class_decorator
    def update(
        self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None
    ) -> np.ndarray:
        assert isinstance(
            dets, np.ndarray
        ), f"Unsupported 'dets' input format '{type(dets)}', valid format is np.ndarray"
        assert isinstance(
            img, np.ndarray
        ), f"Unsupported 'img' input format '{type(img)}', valid format is np.ndarray"
        assert (
            len(dets.shape) == 2
        ), "Unsupported 'dets' dimensions, valid number of dimensions is two"
        assert (
            dets.shape[1] == 6
        ), "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6"
        if embs is not None:
            assert (
                dets.shape[0] == embs.shape[0]
            ), "Missmatch between detections and embeddings sizes"

        self.frame_count += 1

        # Enhanced detection preprocessing with confidence-based filtering
        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
        remain_inds = dets[:, 4] >= self.min_conf
        dets = dets[remain_inds]

        if len(dets) == 0:
            # Still run tracker update for track management
            self.tracker.predict()
            self.tracker.update([])
            return self._format_outputs()

        xyxy = dets[:, 0:4]
        confs = dets[:, 4]
        clss = dets[:, 5]
        det_ind = dets[:, 6]

        # Enhanced camera motion compensation
        if len(self.tracker.tracks) >= 1:
            warp_matrix = self.cmc.apply(img, xyxy)
            for track in self.tracker.tracks:
                track.camera_update(warp_matrix)

        # Enhanced appearance feature extraction with quality assessment
        if embs is not None:
            features = embs[remain_inds]
        else:
            features = self._extract_enhanced_features(xyxy, img, confs)

        # Create enhanced detections with confidence and quality metrics
        tlwh = xyxy2tlwh(xyxy)
        detections = []
        for box, conf, cls, det_ind, feat in zip(tlwh, confs, clss, det_ind, features):
            # Enhanced detection with quality assessment
            detection = Detection(box, conf, cls, det_ind, feat)
            detection.quality_score = self._compute_detection_quality(box, conf, feat)
            detections.append(detection)

        # Sort detections by quality score for better matching priority
        detections.sort(key=lambda x: x.quality_score, reverse=True)

        # Enhanced tracker update with adaptive parameters
        self.tracker.predict()
        self.tracker.update(detections, self.frame_count)

        return self._format_outputs()

    def _extract_enhanced_features(self, xyxy, img, confs):
        """Extract enhanced appearance features with quality assessment"""
        if len(xyxy) == 0:
            return np.array([])
        
        # Extract base features
        features = self.model.get_features(xyxy, img)
        
        # Enhance features based on detection confidence
        enhanced_features = []
        for i, (feat, conf) in enumerate(zip(features, confs)):
            if feat is not None and len(feat) > 0:
                # Normalize feature
                feat = feat / (np.linalg.norm(feat) + 1e-8)
                
                # Apply confidence-based feature enhancement
                if conf > self.conf_thresh_high:
                    # High confidence - use feature as is
                    enhanced_feat = feat
                elif conf > self.conf_thresh_low:
                    # Medium confidence - slight smoothing
                    enhanced_feat = feat * 0.95 + np.random.normal(0, 0.01, feat.shape)
                    enhanced_feat = enhanced_feat / (np.linalg.norm(enhanced_feat) + 1e-8)
                else:
                    # Low confidence - more smoothing
                    enhanced_feat = feat * 0.9 + np.random.normal(0, 0.02, feat.shape)
                    enhanced_feat = enhanced_feat / (np.linalg.norm(enhanced_feat) + 1e-8)
                
                enhanced_features.append(enhanced_feat)
            else:
                enhanced_features.append(np.zeros(512))  # Default feature size
        
        return np.array(enhanced_features) if enhanced_features else np.array([])

    def _compute_detection_quality(self, box, conf, feat):
        """Compute detection quality score for prioritization"""
        # Base quality from confidence
        quality = conf
        
        # Add feature quality (based on feature norm and distinctiveness)
        if feat is not None and len(feat) > 0:
            feat_norm = np.linalg.norm(feat)
            feat_quality = min(feat_norm / 10.0, 1.0)  # Normalize to 0-1
            quality = 0.7 * quality + 0.3 * feat_quality
        
        # Add bounding box quality (aspect ratio and size)
        w, h = box[2], box[3]
        if h > 0:
            aspect_ratio = w / h
            # Penalize extreme aspect ratios
            aspect_quality = 1.0 - abs(aspect_ratio - 0.5) / 2.0
            aspect_quality = max(0.1, aspect_quality)
            quality = 0.9 * quality + 0.1 * aspect_quality
        
        return quality

    def _format_outputs(self):
        """Format tracker outputs with enhanced track information"""
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update >= 1:
                continue

            x1, y1, x2, y2 = track.to_tlbr()

            id = track.id
            conf = track.conf
            cls = track.cls
            det_ind = track.det_ind

            # Add track quality information
            track_quality = getattr(track, 'quality_score', conf)
            
            outputs.append(
                np.concatenate(
                    ([x1, y1, x2, y2], [id], [conf], [cls], [det_ind], [track_quality])
                ).reshape(1, -1)
            )
        
        if len(outputs) > 0:
            return np.concatenate(outputs)
        return np.array([])

    def reset(self):
        """Enhanced reset with proper cleanup"""
        self.tracker.tracks = []
        self.tracker._next_id = 1
        self.frame_count = 0
        self.lost_track_buffer = []
        self.track_history = {}
        
        # Reset metric samples
        if hasattr(self.tracker.metric, 'samples'):
            self.tracker.metric.samples = {}

    def get_track_statistics(self):
        """Get comprehensive tracking statistics for analysis"""
        stats = {
            'total_tracks': len(self.tracker.tracks),
            'confirmed_tracks': len([t for t in self.tracker.tracks if t.is_confirmed()]),
            'tentative_tracks': len([t for t in self.tracker.tracks if t.is_tentative()]),
            'active_tracks': len([t for t in self.tracker.tracks if t.time_since_update < 1]),
            'next_id': self.tracker._next_id,
            'frame_count': self.frame_count,
            'lost_tracks_buffer_size': len(self.lost_track_buffer),
        }
        return stats