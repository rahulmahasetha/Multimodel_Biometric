# fusion/weighted_fusion.py
def fuse(face, finger, palm):
    return 0.4*face + 0.3*finger + 0.3*palm




'''def weighted_fusion(face_score, finger_score, palm_score, weights=(0.4, 0.3, 0.3)):
    """
    Weighted fusion of modality scores
    
    Args:
        face_score: Face verification score (0-1)
        finger_score: Fingerprint verification score (0-1)
        palm_score: Palm verification score (0-1)
        weights: Tuple of (face_weight, finger_weight, palm_weight)
    
    Returns:
        Fused score (0-1)
    """
    # Validate inputs
    if not (0 <= face_score <= 1 and 0 <= finger_score <= 1 and 0 <= palm_score <= 1):
        print(f"⚠️ Warning: Scores should be 0-1, got: {face_score}, {finger_score}, {palm_score}")
    
    # Ensure weights sum to 1
    if abs(sum(weights) - 1.0) > 0.01:
        print(f"⚠️ Weights should sum to 1, got: {sum(weights)}. Normalizing...")
        total = sum(weights)
        weights = (weights[0]/total, weights[1]/total, weights[2]/total)
    
    # Weighted sum
    fused = (face_score * weights[0] + 
             finger_score * weights[1] + 
             palm_score * weights[2])
    
    # Clip to [0, 1]
    return max(0.0, min(1.0, fused))

def adaptive_fusion(face_score, finger_score, palm_score):
    """
    Adaptive fusion based on modality reliability
    Higher scores get higher weights
    """
    # Base weights
    base_weights = (0.4, 0.35, 0.25)
    
    # Adjust based on confidence
    face_weight = base_weights[0] * (0.8 + 0.2 * face_score)
    finger_weight = base_weights[1] * (0.8 + 0.2 * finger_score)
    palm_weight = base_weights[2] * (0.8 + 0.2 * palm_score)
    
    # Normalize
    total = face_weight + finger_weight + palm_weight
    weights = (face_weight/total, finger_weight/total, palm_weight/total)
    
    return weighted_fusion(face_score, finger_score, palm_score, weights)

# Alias for simple usage
fuse = weighted_fusion'''