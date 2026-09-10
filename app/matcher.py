import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# Default Euclidean distance threshold for 128-dimensional face embeddings
DEFAULT_THRESHOLD = 0.50


def euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
    """Calculates Euclidean distance between two vectors."""
    a = np.array(vec1, dtype=np.float32)
    b = np.array(vec2, dtype=np.float32)
    return float(np.linalg.norm(a - b))


def calculate_confidence(distance: float, threshold: float = DEFAULT_THRESHOLD) -> float:
    """
    Converts a distance score into a user-friendly percentage confidence score (0 to 100%).
    Lower distance means higher confidence.
    """
    if distance < 0:
        return 0.0
    
    # Linear-sigmoid mapping calibrated for face-api.js 128-d descriptors
    if distance <= 0.25:
        # Extremely high similarity (same person, similar lighting/angle)
        confidence = 95.0 + (0.25 - distance) * 20.0
    elif distance <= threshold:
        # Confident match within threshold
        confidence = 70.0 + ((threshold - distance) / (threshold - 0.25)) * 25.0
    else:
        # Below threshold (not matched)
        confidence = max(5.0, 70.0 - ((distance - threshold) / threshold) * 70.0)
        
    return float(min(99.9, max(1.0, confidence)))


def find_best_match(
    query_descriptor: List[float],
    users: List[Dict[str, Any]],
    threshold: float = DEFAULT_THRESHOLD
) -> Tuple[Optional[Dict[str, Any]], float, float]:
    """
    Searches through registered users for the closest facial descriptor match.
    
    Returns:
        (matched_user, distance, confidence)
        If no user meets the threshold, matched_user is None.
    """
    if not users:
        return None, 1.0, 0.0
        
    best_user = None
    min_distance = float("inf")
    
    for user in users:
        descriptor = user.get("face_descriptor")
        if not descriptor or len(descriptor) != len(query_descriptor):
            continue
            
        dist = euclidean_distance(query_descriptor, descriptor)
        if dist < min_distance:
            min_distance = dist
            best_user = user
            
    if best_user is not None and min_distance <= threshold:
        confidence = calculate_confidence(min_distance, threshold)
        return best_user, min_distance, confidence
    else:
        confidence = calculate_confidence(min_distance if min_distance != float("inf") else 1.0, threshold)
        return None, min_distance if min_distance != float("inf") else 1.0, confidence
