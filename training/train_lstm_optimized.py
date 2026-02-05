import numpy as np
import json
import os
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Masking, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Lambda, SpatialDropout1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import metrics, regularizers, losses
from preprocessing.mediapipe_extractor import process_video, create_holistic_landmarkers

def augment_landmarks(sequence):
    """Apply random scaling, jitter, temporal scaling, and shifting."""
    # 1. Spatial Scale: 90% to 110%
    scale = np.random.uniform(0.9, 1.1)
    sequence = sequence * scale
    
    # 2. Jitter: small Gaussian noise
    noise = np.random.normal(0, 0.003, sequence.shape)
    mask = (sequence != 0).any(axis=-1, keepdims=True)
    sequence = sequence + (noise * mask)
    
    # 3. Temporal Scaling: change speed of the sign
    # We'll skip frames or interpolate? Simpler: Shift the sequence
    if np.random.random() > 0.5:
        shift = np.random.randint(-5, 5)
        if shift > 0:
            sequence = np.pad(sequence, ((shift, 0), (0, 0)), mode='constant')[:-shift]
        elif shift < 0:
            sequence = np.pad(sequence, ((0, -shift), (0, 0)), mode='constant')[-shift:]
            
    return sequence.astype(np.float32)

def create_lstm_model(num_classes):
    """Create and compile a robust Bidirectional LSTM model with strong regularization."""
    reg = regularizers.l2(0.001) # Stronger regularization
    
    model = Sequential([
        Input(shape=(64, 300)),
        Masking(mask_value=0.0), # Masking FIRST
        
        # Pure RNN stack for cleaner temporal learning on landmarks
        Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=reg)),
        BatchNormalization(),
        Dropout(0.5),
        
        Bidirectional(LSTM(128, kernel_regularizer=reg)),
        BatchNormalization(),
        Dropout(0.5),
        
        # Dense Classifier
        Dense(256, activation="relu", kernel_regularizer=reg),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(num_classes, activation="softmax")
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=5e-4), # Higher initial LR
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=[
            "accuracy",
            metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_accuracy')
        ]
    )
    
    return model

def train_lstm_large_dataset(dataset, num_classes=None, epochs=50, batch_size=32, validation_split=0.2):
    """Train LSTM on large dataset using memory-efficient approach."""
    # 1. Determine classes and filter if requested
    print("🔍 Scanning dataset to determine classes...")
    all_label_info = {item["label"]: item.get("gloss", f"sign_{item['label']}") for item in dataset}
    label_counts = Counter([item["label"] for item in dataset])
    
    if num_classes is not None and num_classes < len(label_counts):
        print(f"🎯 Filtering for TOP {num_classes} most frequent signs...")
        top_signs = [l for l, count in label_counts.most_common(num_classes)]
        dataset = dataset.filter(lambda x: x["label"] in top_signs)
        total_samples = len(dataset)
        print(f"✅ Filtered dataset: {total_samples} samples")
    
    # Create the NEW mapping from old label -> new index [0...num_classes-1]
    final_unique_labels = sorted(list(set([item["label"] for item in dataset])))
    num_classes = len(final_unique_labels)
    
    label_to_idx = {old_label: i for i, old_label in enumerate(final_unique_labels)}
    idx_to_name = {i: all_label_info[old_label] for i, old_label in enumerate(final_unique_labels)}
    
    print(f"✅ Final training on {num_classes} unique classes\n")
    
    # 2. Split for extraction tracking (using the FILTERED dataset)
    split_idx = int(total_samples * (1 - validation_split))
    train_dataset = dataset.select(range(split_idx))
    val_dataset = dataset.select(range(split_idx, total_samples))
    
    # Cache setup
    cache_dir = "dataset/cache"
    os.makedirs(cache_dir, exist_ok=True)
    x_train_path = os.path.join(cache_dir, "X_train_holistic.npy")
    y_train_path = os.path.join(cache_dir, "y_train_holistic.npy")
    x_val_path = os.path.join(cache_dir, "X_val_holistic.npy")
    y_val_path = os.path.join(cache_dir, "y_val_holistic.npy")
    
    if all(os.path.exists(p) for p in [x_train_path, y_train_path, x_val_path, y_val_path]):
        print("📁 Loading pre-processed HOLISTIC features from disk cache...")
        X_train = np.load(x_train_path)
        y_train = np.load(y_train_path)
        X_val = np.load(x_val_path)
        y_val = np.load(y_val_path)
    else:
        print("🔋 Initializing MediaPipe HOLISTIC landmarkers...")
        landmarkers = create_holistic_landmarkers()
        
        print("🎬 Processing training videos (Holistic)...")
        X_train, y_train = [], []
        for i, item in enumerate(train_dataset):
            v_path = item["video"].get("path")
            if v_path:
                try:
                    feat = process_video(v_path, landmarkers=landmarkers)
                    X_train.append(feat)
                    y_train.append(item["label"])
                except: continue
            if (i+1) % 100 == 0: print(f"   Processed {i+1}/{len(train_dataset)} training videos")
        
        print("🎬 Processing validation videos (Holistic)...")
        X_val, y_val = [], []
        for i, item in enumerate(val_dataset):
            v_path = item["video"].get("path")
            if v_path:
                try:
                    feat = process_video(v_path, landmarkers=landmarkers)
                    X_val.append(feat)
                    y_val.append(item["label"])
                except: continue
            if (i+1) % 100 == 0: print(f"   Processed {i+1}/{len(val_dataset)} validation videos")
            
        for l in landmarkers.values(): l.close()
        X_train, y_train, X_val, y_val = np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val)
        np.save(x_train_path, X_train); np.save(y_train_path, y_train)
        np.save(x_val_path, X_val); np.save(y_val_path, y_val)

    # CRITICAL SHUFFLE
    print("🔀 Shuffling and re-splitting data...")
    X_all = np.concatenate([X_train, X_val], axis=0)
    y_all = np.concatenate([y_train, y_val], axis=0)
    
    # Filter arrays to match final_unique_labels (incase cache has extra classes)
    mask = np.isin(y_all, final_unique_labels)
    X_all = X_all[mask]
    y_all = y_all[mask]
    
    print(f"📊 Final dataset size after filtering for selected classes: {len(X_all)}")
    
    idx = np.arange(len(X_all))
    np.random.seed(42); np.random.shuffle(idx)
    X_all, y_all = X_all[idx], y_all[idx]
    
    split = int(len(X_all) * (1 - validation_split))
    X_train, y_train = X_all[:split], y_all[:split]
    X_val, y_val = X_all[split:], y_all[split:]
    
    model = create_lstm_model(num_classes)
    callbacks = [
        ModelCheckpoint('inference/lstm_model_best.keras', monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]
    
    # CRITICAL: Re-map labels to [0, num_classes-1] before final concatenation
    y_train = np.array([label_to_idx[l] for l in y_train])
    y_val = np.array([label_to_idx[l] for l in y_val])

    # SIMPLE AUGMENTATION: Double the training set with augmented versions
    print("🪄 Applying data augmentation (doubling training set)...")
    X_aug = np.array([augment_landmarks(x) for x in X_train])
    X_train_final = np.concatenate([X_train, X_aug], axis=0)
    y_train_final = np.concatenate([y_train, y_train], axis=0)

    # Re-shuffle final training set
    idx = np.arange(len(X_train_final))
    np.random.shuffle(idx)
    X_train_final, y_train_final = X_train_final[idx], y_train_final[idx]
    
    history = model.fit(X_train_final, y_train_final, validation_data=(X_val, y_val), 
                        epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1)
    
    model.save("inference/lstm_model.keras")
    return model, history, idx_to_name

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-run", action="store_true", help="Run a quick smoke test")
    args = parser.parse_args()
    
    if args.test_run:
        print("🧪 RUNNING SMOKE TEST...")
        # Create dummy data
        num_test_classes = 10
        dummy_X = np.random.random((8, 64, 300)).astype(np.float32)
        dummy_y = np.random.randint(0, num_test_classes, 8)
        
        # Compile and train for 1 epoch
        model = create_lstm_model(num_test_classes)
        model.fit(dummy_X, dummy_y, epochs=1, batch_size=2, verbose=1)
        print("✅ SMOKE TEST PASSED: Model compiled and trained for 1 epoch.")
