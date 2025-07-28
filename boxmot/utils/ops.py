# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

from typing import Tuple, Union, List, Optional
import cv2
import numpy as np
import torch
from math import sqrt


def xyxy2xywh(x):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format.

    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.
    Returns:
       y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x, y, width, height) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x_c, y_c, width, height) format to
    (x1, y1, x2, y2) format where (x1, y1) is the top-left corner and (x2, y2)
    is the bottom-right corner.

    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
    Returns:
        y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def xywh2tlwh(x):
    """
    Convert bounding box coordinates from (x c, y c, w, h) format to (t, l, w, h) format where (t, l) is the
    top-left corner and (w, h) is width and height.

    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
    Returns:
        y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (t, l, w, h) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    return y


def tlwh2xyxy(x):
    """
    Convert bounding box coordinates from (t, l, w, h) format to (x1, y1, x2, y2) format where
    (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 2] = x[..., 0] + x[..., 2]  # x2 = x1 + width
    y[..., 3] = x[..., 1] + x[..., 3]  # y2 = y1 + height
    return y


def xyxy2tlwh(x):
    """Convert nx4 boxes from [x1, y1, x2, y2] to [x1, y1, w, h] where xy1=top-left, xy2=bottom-right"""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def tlwh2xywh(x):
    """Convert bounding box coordinates from (t, l, w, h) to (x_c, y_c, w, h) format."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] + x[..., 2] / 2  # center x
    y[..., 1] = x[..., 1] + x[..., 3] / 2  # center y
    return y


def tlwh2xyah(x):
    """
    Convert bounding box coordinates from (top, left, width, height) format to 
    (center_x, center_y, aspect_ratio, height) format.
    
    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (t, l, w, h) format.
    Returns:
        y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x_c, y_c, a, h) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] + x[..., 2] / 2  # center x
    y[..., 1] = x[..., 1] + x[..., 3] / 2  # center y
    y[..., 2] = x[..., 2] / x[..., 3]  # aspect ratio (width / height)
    y[..., 3] = x[..., 3]  # height
    return y


def xyah2tlwh(x):
    """
    Convert bounding box coordinates from (center_x, center_y, aspect_ratio, height) format to
    (top, left, width, height) format.
    
    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x_c, y_c, a, h) format.
    Returns:
        y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (t, l, w, h) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 2] = x[..., 2] * x[..., 3]  # width = aspect_ratio * height
    y[..., 0] = x[..., 0] - y[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 3] = x[..., 3]  # height
    return y


def xyah2xyxy(x):
    """
    Convert bounding box coordinates from (center_x, center_y, aspect_ratio, height) format to
    (x1, y1, x2, y2) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    w = x[..., 2] * x[..., 3]  # width = aspect_ratio * height
    y[..., 0] = x[..., 0] - w / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + w / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y


def xyxy2xyah(x):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to
    (center_x, center_y, aspect_ratio, height) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # center x
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # center y
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    y[..., 2] = (x[..., 2] - x[..., 0]) / y[..., 3]  # aspect ratio
    return y


def compute_box_overlap(box1: np.ndarray, box2: np.ndarray, 
                       format: str = 'xyxy') -> float:
    """
    Compute overlap ratio between two bounding boxes.
    
    Args:
        box1: First bounding box
        box2: Second bounding box
        format: Box format ('xyxy', 'tlwh', 'xywh')
        
    Returns:
        Overlap ratio (intersection / area of box1)
    """
    if format == 'tlwh':
        box1 = tlwh2xyxy(box1)
        box2 = tlwh2xyxy(box2)
    elif format == 'xywh':
        box1 = xywh2xyxy(box1)
        box2 = xywh2xyxy(box2)
    
    # Compute intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    
    return intersection / area1 if area1 > 0 else 0.0


def compute_box_center_distance(box1: np.ndarray, box2: np.ndarray,
                               format: str = 'xyxy',
                               normalize: bool = False,
                               image_size: Tuple[int, int] = None) -> float:
    """
    Compute center-to-center distance between two bounding boxes.
    
    Args:
        box1: First bounding box
        box2: Second bounding box
        format: Box format ('xyxy', 'tlwh', 'xywh')
        normalize: Whether to normalize by image diagonal
        image_size: (width, height) for normalization
        
    Returns:
        Center distance
    """
    if format == 'xyxy':
        c1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
        c2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]
    elif format == 'tlwh':
        c1 = [box1[0] + box1[2] / 2, box1[1] + box1[3] / 2]
        c2 = [box2[0] + box2[2] / 2, box2[1] + box2[3] / 2]
    elif format == 'xywh':
        c1 = [box1[0], box1[1]]
        c2 = [box2[0], box2[1]]
    
    distance = sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
    
    if normalize and image_size:
        diagonal = sqrt(image_size[0]**2 + image_size[1]**2)
        distance /= diagonal
    
    return distance


def expand_box(box: np.ndarray, factor: float = 1.1, 
               format: str = 'xyxy', 
               image_size: Tuple[int, int] = None) -> np.ndarray:
    """
    Expand bounding box by a factor.
    
    Args:
        box: Bounding box coordinates
        factor: Expansion factor (>1 for expansion, <1 for contraction)
        format: Box format ('xyxy', 'tlwh', 'xywh')
        image_size: (width, height) to clip expanded box
        
    Returns:
        Expanded bounding box
    """
    if format == 'xyxy':
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w/2, y1 + h/2
        
        new_w, new_h = w * factor, h * factor
        new_x1 = cx - new_w/2
        new_y1 = cy - new_h/2
        new_x2 = cx + new_w/2
        new_y2 = cy + new_h/2
        
        expanded = np.array([new_x1, new_y1, new_x2, new_y2])
        
    elif format == 'tlwh':
        x, y, w, h = box
        cx, cy = x + w/2, y + h/2
        
        new_w, new_h = w * factor, h * factor
        new_x = cx - new_w/2
        new_y = cy - new_h/2
        
        expanded = np.array([new_x, new_y, new_w, new_h])
        
    elif format == 'xywh':
        cx, cy, w, h = box
        new_w, new_h = w * factor, h * factor
        expanded = np.array([cx, cy, new_w, new_h])
    
    # Clip to image boundaries if provided
    if image_size and format == 'xyxy':
        expanded[0] = max(0, expanded[0])
        expanded[1] = max(0, expanded[1])
        expanded[2] = min(image_size[0], expanded[2])
        expanded[3] = min(image_size[1], expanded[3])
    
    return expanded


def crop_box_region(image: np.ndarray, box: np.ndarray, 
                   format: str = 'xyxy', 
                   expand_factor: float = 1.0) -> np.ndarray:
    """
    Crop image region defined by bounding box.
    
    Args:
        image: Input image
        box: Bounding box coordinates
        format: Box format ('xyxy', 'tlwh', 'xywh')
        expand_factor: Factor to expand box before cropping
        
    Returns:
        Cropped image region
    """
    h, w = image.shape[:2]
    
    if expand_factor != 1.0:
        box = expand_box(box, expand_factor, format, (w, h))
    
    if format == 'tlwh':
        box = tlwh2xyxy(box)
    elif format == 'xywh':
        box = xywh2xyxy(box)
    
    x1, y1, x2, y2 = box.astype(int)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    return image[y1:y2, x1:x2]


def smooth_box_trajectory(boxes: List[np.ndarray], 
                         alpha: float = 0.7,
                         format: str = 'xyxy') -> List[np.ndarray]:
    """
    Apply exponential moving average smoothing to box trajectory.
    
    Args:
        boxes: List of bounding boxes over time
        alpha: Smoothing factor (higher = less smoothing)
        format: Box format
        
    Returns:
        Smoothed trajectory
    """
    if len(boxes) <= 1:
        return boxes
    
    smoothed = [boxes[0]]
    
    for i in range(1, len(boxes)):
        current = boxes[i]
        previous = smoothed[-1]
        
        # Exponential moving average
        smoothed_box = alpha * current + (1 - alpha) * previous
        smoothed.append(smoothed_box)
    
    return smoothed


def compute_motion_vector(box1: np.ndarray, box2: np.ndarray,
                         format: str = 'xyxy') -> np.ndarray:
    """
    Compute motion vector between two bounding boxes.
    
    Args:
        box1: Previous bounding box
        box2: Current bounding box
        format: Box format
        
    Returns:
        Motion vector [dx, dy, dw, dh]
    """
    if format == 'xyxy':
        c1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
        c2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    elif format == 'tlwh':
        c1 = [box1[0] + box1[2] / 2, box1[1] + box1[3] / 2]
        c2 = [box2[0] + box2[2] / 2, box2[1] + box2[3] / 2]
        w1, h1 = box1[2], box1[3]
        w2, h2 = box2[2], box2[3]
    elif format == 'xywh':
        c1, c2 = box1[:2], box2[:2]
        w1, h1 = box1[2], box1[3]
        w2, h2 = box2[2], box2[3]
    
    dx = c2[0] - c1[0]
    dy = c2[1] - c1[1]
    dw = w2 - w1
    dh = h2 - h1
    
    return np.array([dx, dy, dw, dh])


def predict_box_position(box: np.ndarray, motion_vector: np.ndarray,
                        format: str = 'xyxy') -> np.ndarray:
    """
    Predict next box position using motion vector.
    
    Args:
        box: Current bounding box
        motion_vector: Motion vector [dx, dy, dw, dh]
        format: Box format
        
    Returns:
        Predicted bounding box
    """
    dx, dy, dw, dh = motion_vector
    
    if format == 'xyxy':
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w/2, y1 + h/2
        
        new_cx = cx + dx
        new_cy = cy + dy
        new_w = w + dw
        new_h = h + dh
        
        predicted = np.array([
            new_cx - new_w/2, new_cy - new_h/2,
            new_cx + new_w/2, new_cy + new_h/2
        ])
    elif format == 'tlwh':
        x, y, w, h = box
        cx, cy = x + w/2, y + h/2
        
        new_cx = cx + dx
        new_cy = cy + dy
        new_w = w + dw
        new_h = h + dh
        
        predicted = np.array([
            new_cx - new_w/2, new_cy - new_h/2, new_w, new_h
        ])
    elif format == 'xywh':
        cx, cy, w, h = box
        predicted = np.array([cx + dx, cy + dy, w + dw, h + dh])
    
    return predicted


def compute_box_stability(boxes: List[np.ndarray], 
                         format: str = 'xyxy') -> float:
    """
    Compute stability score for a sequence of bounding boxes.
    
    Args:
        boxes: List of bounding boxes over time
        format: Box format
        
    Returns:
        Stability score (0-1, higher is more stable)
    """
    if len(boxes) < 2:
        return 1.0
    
    # Compute motion variations
    motions = []
    for i in range(1, len(boxes)):
        motion = compute_motion_vector(boxes[i-1], boxes[i], format)
        motions.append(motion)
    
    if not motions:
        return 1.0
    
    motions = np.array(motions)
    
    # Compute coefficient of variation for motion
    motion_std = np.std(motions, axis=0)
    motion_mean = np.abs(np.mean(motions, axis=0)) + 1e-8
    cv = np.mean(motion_std / motion_mean)
    
    # Convert to stability score (lower variation = higher stability)
    stability = 1.0 / (1.0 + cv)
    
    return stability


def filter_boxes_by_area(boxes: np.ndarray, 
                        min_area: float = 100,
                        max_area: float = None,
                        format: str = 'xyxy') -> np.ndarray:
    """
    Filter bounding boxes by area constraints.
    
    Args:
        boxes: Array of bounding boxes
        min_area: Minimum area threshold
        max_area: Maximum area threshold (None for no limit)
        format: Box format
        
    Returns:
        Filtered boxes
    """
    if len(boxes) == 0:
        return boxes
    
    if format == 'xyxy':
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    elif format == 'tlwh':
        areas = boxes[:, 2] * boxes[:, 3]
    elif format == 'xywh':
        areas = boxes[:, 2] * boxes[:, 3]
    
    mask = areas >= min_area
    if max_area is not None:
        mask &= areas <= max_area
    
    return boxes[mask]


def filter_boxes_by_aspect_ratio(boxes: np.ndarray,
                                min_ratio: float = 0.1,
                                max_ratio: float = 10.0,
                                format: str = 'xyxy') -> np.ndarray:
    """
    Filter bounding boxes by aspect ratio constraints.
    
    Args:
        boxes: Array of bounding boxes
        min_ratio: Minimum aspect ratio (width/height)
        max_ratio: Maximum aspect ratio
        format: Box format
        
    Returns:
        Filtered boxes
    """
    if len(boxes) == 0:
        return boxes
    
    if format == 'xyxy':
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
    elif format in ['tlwh', 'xywh']:
        widths = boxes[:, 2]
        heights = boxes[:, 3]
    
    # Avoid division by zero
    heights = np.maximum(heights, 1e-6)
    ratios = widths / heights
    
    mask = (ratios >= min_ratio) & (ratios <= max_ratio)
    return boxes[mask]


def compute_occlusion_matrix(boxes: np.ndarray, 
                           format: str = 'xyxy') -> np.ndarray:
    """
    Compute occlusion matrix showing overlap relationships between boxes.
    
    Args:
        boxes: Array of bounding boxes
        format: Box format
        
    Returns:
        Occlusion matrix where element [i,j] is overlap of box i by box j
    """
    n = len(boxes)
    if n == 0:
        return np.array([])
    
    occlusion_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                overlap = compute_box_overlap(boxes[i], boxes[j], format)
                occlusion_matrix[i, j] = overlap
    
    return occlusion_matrix


def detect_box_anomalies(boxes: List[np.ndarray],
                        motion_threshold: float = 100.0,
                        size_change_threshold: float = 0.5,
                        format: str = 'xyxy') -> List[bool]:
    """
    Detect anomalies in box trajectory (sudden jumps, size changes).
    
    Args:
        boxes: List of bounding boxes over time
        motion_threshold: Threshold for sudden motion detection
        size_change_threshold: Threshold for sudden size change
        format: Box format
        
    Returns:
        List of boolean flags indicating anomalies
    """
    if len(boxes) <= 1:
        return [False] * len(boxes)
    
    anomalies = [False]  # First box cannot be anomaly
    
    for i in range(1, len(boxes)):
        current_box = boxes[i]
        previous_box = boxes[i-1]
        
        # Check motion anomaly
        motion = compute_motion_vector(previous_box, current_box, format)
        motion_magnitude = np.linalg.norm(motion[:2])  # Only position
        
        # Check size change anomaly
        if format == 'xyxy':
            prev_area = (previous_box[2] - previous_box[0]) * (previous_box[3] - previous_box[1])
            curr_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        elif format in ['tlwh', 'xywh']:
            prev_area = previous_box[2] * previous_box[3]
            curr_area = current_box[2] * current_box[3]
        
        size_change_ratio = abs(curr_area - prev_area) / (prev_area + 1e-8)
        
        # Detect anomaly
        is_anomaly = (motion_magnitude > motion_threshold or 
                     size_change_ratio > size_change_threshold)
        anomalies.append(is_anomaly)
    
    return anomalies


def interpolate_missing_boxes(boxes: List[Optional[np.ndarray]],
                             method: str = 'linear',
                             format: str = 'xyxy') -> List[np.ndarray]:
    """
    Interpolate missing bounding boxes in trajectory.
    
    Args:
        boxes: List of boxes with None for missing frames
        method: Interpolation method ('linear', 'spline')
        format: Box format
        
    Returns:
        Complete trajectory with interpolated boxes
    """
    # Find valid (non-None) boxes
    valid_indices = [i for i, box in enumerate(boxes) if box is not None]
    
    if len(valid_indices) < 2:
        # Cannot interpolate with less than 2 valid boxes
        return [box for box in boxes if box is not None]
    
    result = boxes.copy()
    
    # Linear interpolation between consecutive valid boxes
    for i in range(len(valid_indices) - 1):
        start_idx = valid_indices[i]
        end_idx = valid_indices[i + 1]
        
        if end_idx - start_idx > 1:  # There are missing boxes to interpolate
            start_box = boxes[start_idx]
            end_box = boxes[end_idx]
            
            # Interpolate each missing frame
            for j in range(start_idx + 1, end_idx):
                t = (j - start_idx) / (end_idx - start_idx)
                interpolated_box = (1 - t) * start_box + t * end_box
                result[j] = interpolated_box
    
    return [box for box in result if box is not None]


def normalize_boxes(boxes: np.ndarray, 
                   image_size: Tuple[int, int],
                   format: str = 'xyxy') -> np.ndarray:
    """
    Normalize box coordinates to [0, 1] range.
    
    Args:
        boxes: Array of bounding boxes
        image_size: (width, height) of image
        format: Box format
        
    Returns:
        Normalized boxes
    """
    if len(boxes) == 0:
        return boxes
    
    width, height = image_size
    normalized = boxes.copy().astype(float)
    
    if format == 'xyxy':
        normalized[:, [0, 2]] /= width   # x coordinates
        normalized[:, [1, 3]] /= height  # y coordinates
    elif format == 'tlwh':
        normalized[:, 0] /= width   # x coordinate
        normalized[:, 1] /= height  # y coordinate
        normalized[:, 2] /= width   # width
        normalized[:, 3] /= height  # height
    elif format == 'xywh':
        normalized[:, 0] /= width   # center x
        normalized[:, 1] /= height  # center y
        normalized[:, 2] /= width   # width
        normalized[:, 3] /= height  # height
    
    return normalized


def denormalize_boxes(boxes: np.ndarray,
                     image_size: Tuple[int, int],
                     format: str = 'xyxy') -> np.ndarray:
    """
    Denormalize box coordinates from [0, 1] range.
    
    Args:
        boxes: Array of normalized bounding boxes
        image_size: (width, height) of image
        format: Box format
        
    Returns:
        Denormalized boxes
    """
    if len(boxes) == 0:
        return boxes
    
    width, height = image_size
    denormalized = boxes.copy().astype(float)
    
    if format == 'xyxy':
        denormalized[:, [0, 2]] *= width   # x coordinates
        denormalized[:, [1, 3]] *= height  # y coordinates
    elif format == 'tlwh':
        denormalized[:, 0] *= width   # x coordinate
        denormalized[:, 1] *= height  # y coordinate
        denormalized[:, 2] *= width   # width
        denormalized[:, 3] *= height  # height
    elif format == 'xywh':
        denormalized[:, 0] *= width   # center x
        denormalized[:, 1] *= height  # center y
        denormalized[:, 2] *= width   # width
        denormalized[:, 3] *= height  # height
    
    return denormalized


# Legacy support for existing functions
def make_divisible(x, divisor):
    """Returns x evenly divisible by divisor"""
    return int(np.ceil(x / divisor) * divisor)


def clip_coords(boxes, img_shape):
    """Clip bounding xyxy bounding boxes to image shape (height, width)"""
    if isinstance(boxes, torch.Tensor):
        boxes[:, 0].clamp_(0, img_shape[1])  # x1
        boxes[:, 1].clamp_(0, img_shape[0])  # y1
        boxes[:, 2].clamp_(0, img_shape[1])  # x2
        boxes[:, 3].clamp_(0, img_shape[0])  # y2
    else:
        boxes[:, 0] = np.clip(boxes[:, 0], 0, img_shape[1])  # x1
        boxes[:, 1] = np.clip(boxes[:, 1], 0, img_shape[0])  # y1
        boxes[:, 2] = np.clip(boxes[:, 2], 0, img_shape[1])  # x2
        boxes[:, 3] = np.clip(boxes[:, 3], 0, img_shape[0])  # y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """Rescale coords (xyxy) from img1_shape to img0_shape"""
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords
