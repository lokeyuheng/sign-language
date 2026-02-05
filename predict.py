import numpy as np
import json
import os
from tensorflow.keras.models import load_model
from preprocessing.mediapipe_extractor import process_video, process_frames
from preprocessing.video_segmentation import segment_video_by_motion, segment_video_fixed_window, extract_segment


class SignLanguagePredictor:
    """Predictor for sign language recognition using trained LSTM model."""
    
    def __init__(self, model_path="inference/lstm_model_best.keras", label_map_path="inference/label_map.json"):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to trained LSTM model (.keras or .h5)
            label_map_path: Path to label mapping JSON
        """
        print(f"Loading model from {model_path}...")
        
        # Try .keras first, fallback to .h5 for backward compatibility
        if not os.path.exists(model_path):
            # Try alternative extension
            if model_path.endswith('.keras'):
                alt_path = model_path.replace('.keras', '.h5')
            else:
                alt_path = model_path.replace('.h5', '.keras')
            
            if os.path.exists(alt_path):
                print(f"  Model not found, trying {alt_path}...")
                model_path = alt_path
        
        self.model = load_model(model_path)
        
        print(f"Loading label map from {label_map_path}...")
        with open(label_map_path, 'r') as f:
            # Load and convert string keys to int
            label_map_str = json.load(f)
            self.label_map = {int(k): v for k, v in label_map_str.items()}
        
        print(f"✅ Model loaded with {len(self.label_map)} sign classes")
    
    def predict_sign(self, video_path):
        """
        Predict a single sign from a video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            dict: Contains 'label_id', 'sign_text', 'confidence'
        """
        # Extract features from video
        features = process_video(video_path, max_frames=64)
        
        # Add batch dimension: (1, 64, 300)
        features = np.expand_dims(features, axis=0)
        
        # Predict
        predictions = self.model.predict(features, verbose=0)
        
        # Get predicted class and confidence
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        # Get sign text from label map
        sign_text = self.label_map.get(predicted_class, f"UNKNOWN_{predicted_class}")
        
        return {
            "label_id": int(predicted_class),
            "sign_text": sign_text,
            "confidence": confidence,
            "all_probabilities": predictions[0].tolist()
        }
    
    def predict_from_frames(self, frames):
        """
        Predict a sign from a list of frames.
        
        Args:
            frames: List of BGR frames
            
        Returns:
            dict: Contains 'label_id', 'sign_text', 'confidence'
        """
        # Extract features from frames
        features = process_frames(frames, max_frames=64)
        
        # Add batch dimension: (1, 64, 300)
        features = np.expand_dims(features, axis=0)
        
        # Predict
        predictions = self.model.predict(features, verbose=0)
        
        # Get predicted class and confidence
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        # Get sign text from label map
        sign_text = self.label_map.get(predicted_class, f"UNKNOWN_{predicted_class}")
        
        return {
            "label_id": int(predicted_class),
            "sign_text": sign_text,
            "confidence": confidence
        }
    
    def predict_sequence(self, video_path, method="motion", confidence_threshold=0.3):
        """
        Predict multiple signs from a video containing a sequence.
        
        Args:
            video_path: Path to video file
            method: Segmentation method - "motion" or "fixed"
            confidence_threshold: Minimum confidence to include a prediction
            
        Returns:
            dict: Contains 'signs' (list of predictions) and 'raw_sequence'
        """
        print(f"🎬 Segmenting video using '{method}' method...")
        
        # Segment the video
        if method == "motion":
            segments = segment_video_by_motion(
                video_path,
                min_segment_frames=20,
                max_segment_frames=100,
                motion_threshold=0.02,
                pause_frames=10
            )
        else:  # fixed window
            segments = segment_video_fixed_window(
                video_path,
                window_size=64,
                overlap=16
            )
        
        print(f"   Found {len(segments)} segments")
        
        # Predict each segment
        predictions = []
        for i, (start, end) in enumerate(segments):
            print(f"   Processing segment {i+1}/{len(segments)}: frames {start}-{end}")
            
            # Extract frames for this segment
            frames = extract_segment(video_path, start, end)
            
            if len(frames) < 10:  # Skip very short segments
                continue
            
            # Predict
            result = self.predict_from_frames(frames)
            
            # Only include predictions above confidence threshold
            if result['confidence'] >= confidence_threshold:
                predictions.append({
                    "sign_text": result["sign_text"],
                    "confidence": result["confidence"],
                    "segment": (start, end)
                })
        
        # Extract just the sign sequence
        sign_sequence = [p["sign_text"] for p in predictions]
        
        return {
            "signs": predictions,
            "sign_sequence": sign_sequence,
            "num_segments": len(segments),
            "num_signs_detected": len(predictions)
        }
    
    def predict_top_k(self, video_path, k=5):
        """
        Get top K predictions for a video.
        
        Args:
            video_path: Path to video file
            k: Number of top predictions to return
            
        Returns:
            list: Top K predictions with sign text and confidence
        """
        # Extract features
        features = process_video(video_path, max_frames=64)
        features = np.expand_dims(features, axis=0)
        
        # Predict
        predictions = self.model.predict(features, verbose=0)[0]
        
        # Get top K indices
        top_k_indices = np.argsort(predictions)[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            results.append({
                "label_id": int(idx),
                "sign_text": self.label_map.get(idx, f"UNKNOWN_{idx}"),
                "confidence": float(predictions[idx])
            })
        
        return results


def predict_from_video(video_path, model_path="inference/lstm_model_best.keras", label_map_path="inference/label_map.json"):
    """
    Convenience function to predict from a video without creating a predictor instance.
    
    Args:
        video_path: Path to video file
        model_path: Path to trained model (.keras or .h5)
        label_map_path: Path to label mapping
        
    Returns:
        dict: Prediction result
    """
    predictor = SignLanguagePredictor(model_path, label_map_path)
    return predictor.predict_sign(video_path)
    # No changes needed here, just verifying consistency.
