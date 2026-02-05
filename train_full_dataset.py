import json
import os
import logging

# Suppress TensorFlow and MediaPipe logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=no info, 2=no info/warnings, 3=no info/warnings/errors
os.environ['GLOG_minloglevel'] = '2'       # Suppress MediaPipe/C++ logs (2 = ERROR)

from dataset.download_wlasl import load_wlasl
from training.train_lstm_optimized import train_lstm_large_dataset

print("=" * 60)
print("🎯 TRAINING ON FULL WLASL DATASET")
print("=" * 60)

# Load the FULL dataset (no max_samples limit)
print("\n📥 Loading WLASL dataset from HuggingFace...")
dataset = load_wlasl(split="train")  # ← No limit = ALL videos!

print(f"\n✅ Loaded {len(dataset)} videos")

# Optional: Filter by vocabulary subset for specific signs
# Uncomment and modify if you want to train on specific signs only:
# vocab_subset = [0, 1, 2, 3, 4]  # Train on first 5 sign classes only
# dataset = dataset.filter(lambda x: x["label"] in vocab_subset)

# Train the model
# NOTE: Setting num_classes = 100 will filter for the top 100 most frequent signs
# This significantly improves accuracy by providing more samples per class.
# Train the model
# NOTE: Setting num_classes = 100 will filter for the top 100 most frequent signs
# This significantly improves accuracy by providing more samples per class.
model, history, label_map = train_lstm_large_dataset(
    dataset=dataset,
    num_classes=100,  # Focus on top 100 signs for better accuracy
    epochs=100,       # More epochs for the complex Bidirectional model
    batch_size=32,    # Increased for stability
    validation_split=0.2
)

# Save label mapping for inference
with open("inference/label_map.json", "w") as f:
    json.dump(label_map, f, indent=2)

print("\n🎉 Training complete!")
print("✅ Model saved to: inference/lstm_model.keras")
print("✅ Best model saved to: inference/lstm_model_best.keras")
print("✅ Label map saved to: inference/label_map.json")
