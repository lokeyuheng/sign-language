import cv2
import numpy as np
from collections import deque


def detect_motion(frame1, frame2, threshold=25):
    """
    Detect motion between two frames using frame differencing.
    
    Args:
        frame1: Previous frame (grayscale)
        frame2: Current frame (grayscale)
        threshold: Pixel difference threshold
        
    Returns:
        float: Motion score (0-1, higher = more motion)
    """
    diff = cv2.absdiff(frame1, frame2)
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    motion_pixels = np.sum(thresh > 0)
    total_pixels = thresh.shape[0] * thresh.shape[1]
    return motion_pixels / total_pixels


def segment_video_by_motion(video_path, min_segment_frames=30, max_segment_frames=120, 
                            motion_threshold=0.02, pause_frames=15):
    """
    Segment a video into individual signs based on motion detection.
    
    Strategy:
    1. Detect periods of high motion (signing)
    2. Detect periods of low/no motion (pauses between signs)
    3. Split video at pause boundaries
    
    Args:
        video_path: Path to input video
        min_segment_frames: Minimum frames for a valid segment
        max_segment_frames: Maximum frames for a segment
        motion_threshold: Motion score threshold (lower = more sensitive)
        pause_frames: Number of consecutive low-motion frames to detect a pause
        
    Returns:
        list: List of (start_frame, end_frame) tuples for each segment
    """
    cap = cv2.VideoCapture(video_path)
    
    segments = []
    motion_history = deque(maxlen=pause_frames)
    
    prev_frame = None
    frame_idx = 0
    segment_start = 0
    in_sign = False
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale for motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate motion if we have a previous frame
        if prev_frame is not None:
            motion = detect_motion(prev_frame, gray)
            motion_history.append(motion)
            
            # Check if we're in a pause (low motion)
            is_pause = len(motion_history) == pause_frames and \
                      np.mean(motion_history) < motion_threshold
            
            if not in_sign and not is_pause:
                # Start of new sign detected
                segment_start = frame_idx
                in_sign = True
            
            elif in_sign and is_pause:
                # End of sign detected (pause)
                segment_length = frame_idx - segment_start
                
                if segment_length >= min_segment_frames:
                    segments.append((segment_start, frame_idx))
                
                in_sign = False
            
            elif in_sign and (frame_idx - segment_start) >= max_segment_frames:
                # Force segment split if too long
                segments.append((segment_start, frame_idx))
                segment_start = frame_idx
        
        prev_frame = gray
        frame_idx += 1
    
    # Handle last segment if video ends during a sign
    if in_sign and (frame_idx - segment_start) >= min_segment_frames:
        segments.append((segment_start, frame_idx))
    
    cap.release()
    
    return segments


def segment_video_fixed_window(video_path, window_size=64, overlap=16):
    """
    Segment video using fixed-size sliding windows (simpler approach).
    
    Args:
        video_path: Path to input video
        window_size: Number of frames per segment
        overlap: Number of overlapping frames between segments
        
    Returns:
        list: List of (start_frame, end_frame) tuples for each segment
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    segments = []
    step = window_size - overlap
    
    for start in range(0, total_frames - window_size + 1, step):
        end = start + window_size
        segments.append((start, end))
    
    return segments


def extract_segment(video_path, start_frame, end_frame):
    """
    Extract a segment from a video.
    
    Args:
        video_path: Path to video
        start_frame: Starting frame index
        end_frame: Ending frame index
        
    Returns:
        list: List of frames (BGR format)
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames = []
    for i in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames
