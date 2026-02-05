import numpy as np

def normalize_pose(pose_pts):
    if pose_pts is None or len(pose_pts) < 2:
        return np.zeros(12)
    left_shoulder, right_shoulder = pose_pts[0], pose_pts[1]
    anchor = (left_shoulder + right_shoulder) / 2
    scale = np.linalg.norm(left_shoulder - right_shoulder) + 1e-6
    return ((pose_pts - anchor) / scale).flatten()

def normalize_hand(hand_pts):
    if hand_pts is None or len(hand_pts) == 0:
        return np.zeros(21*3)
    wrist = hand_pts[0]
    return (hand_pts - wrist).flatten()
