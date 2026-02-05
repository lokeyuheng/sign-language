import numpy as np
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Masking, Bidirectional, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import metrics
from preprocessing.mediapipe_extractor import process_video

def train_lstm(video_paths, labels, num_classes):
    """
    Train LSTM model from local video paths.
    
    Args:
        video_paths: List of video file paths
        labels: List of corresponding labels
        num_classes: Total number of classes
    """
    X = []
    y = []

    for path, label in zip(video_paths, labels):
        X.append(process_video(path))
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    model = Sequential([
        Input(shape=(64, 258)),
        Masking(mask_value=0.0),
        BatchNormalization(),
        Bidirectional(LSTM(128)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
    )

    model.fit(X, y, epochs=10, batch_size=4)

    model.save("inference/lstm_model.keras")
    print("✅ Model saved to inference/lstm_model.keras")

    return model


def train_lstm_from_dataset(dataset, num_classes=None, epochs=10, batch_size=4):
    """
    Train LSTM model from HuggingFace dataset (e.g., WLASL).
    
    Args:
        dataset: HuggingFace dataset with 'video' and 'label' columns
        num_classes: Total number of classes (auto-detected if None)
        epochs: Number of training epochs
        batch_size: Batch size for training
    """
    X = []
    y = []
    
    print(f"Processing {len(dataset)} videos from dataset...")
    
    for i, item in enumerate(dataset):
        # The video column contains a dict with 'path' since decode=False
        video_path = item["video"].get("path")
        label = item["label"]
        
        if not video_path:
            print(f"⚠️  Skipping item {i}: no video path found")
            continue
        
        try:
            features = process_video(video_path)
            X.append(features)
            y.append(label)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(dataset)} videos...")
                
        except Exception as e:
            print(f"⚠️  Error processing video {video_path}: {e}")
            continue
    
    X = np.array(X)
    y = np.array(y)
    
    # Auto-detect number of classes if not provided
    if num_classes is None:
        num_classes = len(np.unique(y))
        print(f"Auto-detected {num_classes} classes")
    
    print(f"\n📊 Training on {len(X)} videos with {num_classes} classes")
    
    model = Sequential([
        Input(shape=(64, 258)),
        Masking(mask_value=0.0),
        BatchNormalization(),
        Bidirectional(LSTM(128)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])
    
    model.compile(
        optimizer=Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
    )
    
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    
    model.save("inference/lstm_model.keras")
    print("✅ Model saved to inference/lstm_model.keras")
    
    return model
