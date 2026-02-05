from dataset.download_wlasl import load_wlasl
from training.train_lstm import train_lstm_from_dataset

# Load a small subset from HuggingFace WLASL dataset
print("📥 Loading WLASL dataset from HuggingFace...")
dataset = load_wlasl(split="train", max_samples=50)  # Start with 50 videos

print(f"✅ Loaded {len(dataset)} videos")
print(f"Dataset columns: {dataset.column_names}")
print(f"First item: {dataset[0]}")

# Train the LSTM model
model = train_lstm_from_dataset(
    dataset=dataset,
    epochs=20,  # Increase epochs for better learning
    batch_size=8
)

print("\n🎉 Training complete!")
