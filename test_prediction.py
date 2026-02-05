"""
Test script to verify the complete inference pipeline.
Run this after training your model to ensure everything works.
"""
from inference.predict import SignLanguagePredictor
import sys

def test_prediction(video_path):
    """Test prediction on a single video."""
    print("=" * 60)
    print("🧪 TESTING SIGN LANGUAGE PREDICTION")
    print("=" * 60)
    
    # Initialize predictor
    print("\n📦 Loading model...")
    try:
        predictor = SignLanguagePredictor(
            model_path="inference/lstm_model_best.keras",
            label_map_path="inference/label_map.json"
        )
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 You need to train a model first!")
        print("   Run: python train_full_dataset.py")
        return
    
    # Make prediction
    print(f"\n🎬 Analyzing video: {video_path}")
    result = predictor.predict_sign(video_path)
    
    # Display results
    print("\n" + "=" * 60)
    print("📊 PREDICTION RESULTS")
    print("=" * 60)
    print(f"✅ Predicted Sign: {result['sign_text']}")
    print(f"📈 Confidence: {result['confidence']:.2%}")
    print(f"🔢 Label ID: {result['label_id']}")
    
    # Show top 5 predictions
    print("\n📋 Top 5 Predictions:")
    top_5 = predictor.predict_top_k(video_path, k=5)
    for i, pred in enumerate(top_5, 1):
        print(f"   {i}. {pred['sign_text']:20s} ({pred['confidence']:.2%})")
    
    # Generate sentence (for single sign, it's just the sign itself)
    sentence = result['sign_text']
    print(f"\n💬 Generated Sentence: \"{sentence}\"")
    
    print("\n" + "=" * 60)
    print("✅ Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    # Use video from command line or default
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        # Default test video
        video_path = "uploads/01073.mp4"
    
    test_prediction(video_path)
