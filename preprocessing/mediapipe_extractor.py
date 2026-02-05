import cv2
import mediapipe as mp
import numpy as np
import os

from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HAND_MODEL = os.path.join(BASE_DIR, "hand_landmarker.task")
POSE_MODEL = os.path.join(BASE_DIR, "pose_landmarker.task")
FACE_MODEL = os.path.join(BASE_DIR, "face_landmarker.task")

FEATURE_DIM = 300 

FACE_INDICES = [
    33, 133, 362, 263, 61, 291, 0, 17, 13, 14, 78, 308,
    10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 152, 107
]

def smooth_landmarks(sequence, alpha=0.6):
    """Apply Exponential Moving Average (EMA) smoothing to reduce jitter."""
    if len(sequence) < 2: return sequence
    smoothed = np.copy(sequence)
    for i in range(1, len(sequence)):
        # Only smooth if frame is not zero-padded
        if not np.all(sequence[i] == 0):
            # If previous frame was padding, don't average with it
            if np.all(smoothed[i-1] == 0):
                smoothed[i] = sequence[i]
            else:
                smoothed[i] = alpha * sequence[i] + (1 - alpha) * smoothed[i-1]
    return smoothed

def create_holistic_landmarkers():
    """Create all three landmarkers in IMAGE mode with higher confidence."""
    common_base = lambda p: BaseOptions(model_asset_path=p)
    
    # Increase confidence thresholds to reduce jitter/noise
    hand_opts = HandLandmarkerOptions(
        base_options=common_base(HAND_MODEL), 
        running_mode=mp.tasks.vision.RunningMode.IMAGE, 
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6
    )
    pose_opts = PoseLandmarkerOptions(
        base_options=common_base(POSE_MODEL), 
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6
    )
    face_opts = FaceLandmarkerOptions(
        base_options=common_base(FACE_MODEL), 
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        min_face_detection_confidence=0.6,
        min_face_presence_confidence=0.6,
        min_tracking_confidence=0.6
    )
    
    return {
        "hand": HandLandmarker.create_from_options(hand_opts),
        "pose": PoseLandmarker.create_from_options(pose_opts),
        "face": FaceLandmarker.create_from_options(face_opts)
    }

def extract_holistic_landmarks(frame, landmarkers):
    """Internal helper to extract features with scale-normalization."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # 1. Hands (126 features)
    hand_res = landmarkers["hand"].detect(mp_image)
    left_h = [0.0] * 63; right_h = [0.0] * 63
    if hand_res.hand_landmarks and hand_res.handedness:
        for hl, hd in zip(hand_res.hand_landmarks, hand_res.handedness):
            res = []
            wrist = hl[0]
            # Hand scale normalization: distance between wrist (0) and middle finger MCP (9)
            mcp = hl[9]
            hand_scale = np.sqrt((wrist.x-mcp.x)**2 + (wrist.y-mcp.y)**2 + (wrist.z-mcp.z)**2)
            hand_scale = max(hand_scale, 0.01) # Avoid division by zero
            
            for lm in hl: 
                res.extend([(lm.x - wrist.x)/hand_scale, (lm.y - wrist.y)/hand_scale, (lm.z - wrist.z)/hand_scale])
            
            if hd[0].category_name == 'Left': left_h = res
            else: right_h = res

    # 2. Pose (99 features)
    pose_res = landmarkers["pose"].detect(mp_image)
    pose_f = [0.0] * 99
    if pose_res.pose_landmarks:
        pl = pose_res.pose_landmarks[0]
        # Pose scale normalization: distance between shoulders (11 and 12)
        l_sh, r_sh = pl[11], pl[12]
        shoulder_dist = np.sqrt((l_sh.x-r_sh.x)**2 + (l_sh.y-r_sh.y)**2 + (l_sh.z-r_sh.z)**2)
        pose_scale = max(shoulder_dist, 0.05)
        
        nose = pl[0] 
        res = []
        for lm in pl: 
            res.extend([(lm.x - nose.x)/pose_scale, (lm.y - nose.y)/pose_scale, (lm.z - nose.z)/pose_scale])
        pose_f = res

    # 3. Face (75 features)
    face_res = landmarkers["face"].detect(mp_image)
    face_f = [0.0] * 75
    if face_res.face_landmarks:
        fl = face_res.face_landmarks[0]
        # Scale-normalization: Divide by face height (forehead-to-chin distance)
        top, bottom = fl[10], fl[152]
        face_h = np.sqrt((top.x-bottom.x)**2 + (top.y-bottom.y)**2 + (top.z-bottom.z)**2)
        face_h = max(face_h, 0.05) 
        
        nose = fl[1]
        res = []
        for idx in FACE_INDICES:
            lm = fl[idx]
            res.extend([(lm.x - nose.x)/face_h, (lm.y - nose.y)/face_h, (lm.z - nose.z)/face_h])
        face_f = res

    return left_h + right_h + pose_f + face_f

def process_video(video_path, max_frames=64, landmarkers=None):
    """Extract holistic landmarks from a video file."""
    close_when_done = False
    if landmarkers is None:
        landmarkers = create_holistic_landmarkers()
        close_when_done = True
    
    cap = cv2.VideoCapture(video_path)
    sequence = []
    while cap.isOpened() and len(sequence) < max_frames:
        ret, frame = cap.read()
        if not ret: break
        sequence.append(extract_holistic_landmarks(frame, landmarkers))
    cap.release()
    
    if close_when_done:
        for l in landmarkers.values(): l.close()

    if len(sequence) > 0:
        sequence = smooth_landmarks(np.array(sequence))

    while len(sequence) < max_frames: sequence.append([0.0] * FEATURE_DIM)
    return np.array(sequence, dtype=np.float32)

def process_frames(frames, max_frames=64):
    """Extract holistic landmarks from a list of BGR frames."""
    landmarkers = create_holistic_landmarkers()
    sequence = [extract_holistic_landmarks(f, landmarkers) for f in frames[:max_frames]]
    for l in landmarkers.values(): l.close()

    if len(sequence) > 0:
        sequence = smooth_landmarks(np.array(sequence))

    while len(sequence) < max_frames: sequence.append([0.0] * FEATURE_DIM)
    return np.array(sequence, dtype=np.float32)
