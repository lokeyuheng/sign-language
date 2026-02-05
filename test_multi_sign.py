"""
Test multi-sign sentence prediction.
Run this after training to test the complete sentence generation pipeline.
"""
from inference.predict import SignLanguagePredictor
from inference.gemini_api import generate_sentence_from_signs
import sys

def test_multi_sign_prediction(video_path, method="motion"):
    """Test multi-sign prediction on a video."""
    print("=" * 70)
    print("🎯 TESTING MULTI-SIGN SENTENCE PREDICTION")
    print("=" * 70)
    
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
    
    # Predict sequence
    print(f"\n🎬 Analyzing video: {video_path}")
    print(f"   Segmentation method: {method}")
    
    result = predictor.predict_sequence(
        video_path,
        method=method,
        confidence_threshold=0.3
    )
    
    # Display results
    print("\n" + "=" * 70)
    print("📊 SEGMENTATION RESULTS")
    print("=" * 70)
    print(f"Total segments found: {result['num_segments']}")
    print(f"Signs detected: {result['num_signs_detected']}")
    
    if result['num_signs_detected'] > 0:
        print("\n📋 Detected Signs:")
        for i, sign_info in enumerate(result['signs'], 1):
            print(f"   {i}. {sign_info['sign_text']:20s} "
                  f"(confidence: {sign_info['confidence']:.2%}, "
                  f"frames: {sign_info['segment'][0]}-{sign_info['segment'][1]})")
        
        # Show sign sequence
        print(f"\n🔤 Sign Sequence: {' → '.join(result['sign_sequence'])}")
        
        # Generate sentence
        print("\n" + "=" * 70)
        print("💬 SENTENCE GENERATION")
        print("=" * 70)
        print("Calling Gemini API...")
        
        sentence = generate_sentence_from_signs(result['sign_sequence'])
        
        print(f"\n✨ Generated Sentence:")
        print(f'   "{sentence}"')
        
    else:
        print("\n⚠️  No signs detected in the video.")
        print("   Try adjusting confidence_threshold or use a different video.")
    
    print("\n" + "=" * 70)
    print("✅ Test Complete!")
    print("=" * 70)

if __name__ == "__main__":
    # Get parameters from command line
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        method = sys.argv[2] if len(sys.argv) > 2 else "motion"
    else:
        # Default test video
        video_path = "uploads/01073.mp4"
        method = "motion"
    
    print(f"\nUsage: python test_multi_sign.py <video_path> [motion|fixed]")
    print(f"Using: {video_path} with method={method}\n")
    
    test_multi_sign_prediction(video_path, method)
