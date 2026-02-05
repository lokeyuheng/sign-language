# Sign Language Recognition App - Multi-Sign Sentence Prediction

## 🎯 Features

### Two Prediction Modes

1. **Single Sign Mode** (`/predict`)
   - Detects ONE sign from a video
   - Returns the sign word and confidence
   
2. **Multi-Sign Sequence Mode** (`/predict-sequence`) ⭐ NEW!
   - Segments video into multiple signs
   - Detects each sign separately
   - Uses Gemini AI to generate grammatical sentences

## 🚀 Quick Start

### 1. Train the Model
```bash
python train_full_dataset.py
```
This will:
- Download ~21,000 videos from WLASL dataset
- Train LSTM model (takes several hours)
- Save `lstm_model_best.h5` and `label_map.json`

### 2. Install Gemini API
```bash
pip install google-generativeai
```

### 3. Run the App
```bash
uvicorn app:app --reload
```
Visit: `http://localhost:8000`

## 📡 API Endpoints

### Single Sign Prediction
```bash
POST /predict
```
Upload a video with ONE sign.

**Response:**
```json
{
  "mode": "single",
  "predicted_sign": "hello",
  "confidence": 0.87,
  "sentence": "hello",
  "top_predictions": [...]
}
```

### Multi-Sign Sequence Prediction ⭐
```bash
POST /predict-sequence
Form data:
  - video: file
  - method: "motion" or "fixed"
```

**Response:**
```json
{
  "mode": "sequence",
  "sign_sequence": ["I", "love", "you"],
  "sentence": "I love you",
  "num_segments": 5,
  "num_signs_detected": 3,
  "signs": [
    {"sign_text": "I", "confidence": 0.92, "segment": [0, 45]},
    {"sign_text": "love", "confidence": 0.88, "segment": [60, 105]},
    {"sign_text": "you", "confidence": 0.85, "segment": [120, 165]}
  ]
}
```

## 🧪 Testing

### Test Single Sign
```bash
python test_prediction.py uploads/video.mp4
```

### Test Multi-Sign
```bash
python test_multi_sign.py uploads/video.mp4 motion
```

## 🎬 How Multi-Sign Works

1. **Video Segmentation**
   - **Motion-based**: Detects pauses between signs
   - **Fixed-window**: Splits video into equal segments

2. **Sign Detection**
   - Each segment is processed by MediaPipe
   - LSTM model predicts the sign
   - Only confident predictions (>30%) are kept

3. **Sentence Generation**
   - Sign sequence: `["I", "want", "water"]`
   - Gemini AI converts to: `"I want water"`
   - Adds proper grammar and natural flow

## 📊 Segmentation Methods

### Motion-based (Recommended)
- Detects natural pauses between signs
- Works best for videos with clear separations
- More accurate but slower

### Fixed-window
- Splits video into equal chunks
- Faster but may cut signs in the middle
- Good for continuous signing

## ⚙️ Configuration

Edit parameters in `app.py`:
```python
predictor.predict_sequence(
    video_path,
    method="motion",           # or "fixed"
    confidence_threshold=0.3   # adjust sensitivity
)
```

## 📝 Example Use Cases

1. **Single word queries**: "What does this sign mean?"
2. **Short sentences**: "I need help"
3. **Conversations**: "How are you today?"
4. **Education**: Learning sign language phrases

## 🔧 Troubleshooting

### No signs detected
- Lower `confidence_threshold` (e.g., 0.2)
- Try different `method` (motion vs fixed)
- Ensure video has clear hand movements

### Gemini API errors
- Check API key in `inference/gemini_api.py`
- Verify internet connection
- Falls back to simple word joining

### Poor sentence quality
- Train model with more data
- Use videos with clear signs
- Check if signs are in training vocabulary

## 📚 File Structure

```
sign_recognition_app/
├── preprocessing/
│   ├── mediapipe_extractor.py     # Hand landmark extraction
│   └── video_segmentation.py       # Video segmentation ⭐
├── training/
│   ├── train_lstm.py               # LSTM training
│   └── train_lstm_optimized.py     # Large dataset training
├── inference/
│   ├── predict.py                  # Multi-sign predictor ⭐
│   └── gemini_api.py               # Sentence generation ⭐
├── app.py                          # FastAPI web server ⭐
├── train_full_dataset.py           # Full WLASL training
├── test_prediction.py              # Single sign test
└── test_multi_sign.py              # Multi-sign test ⭐
```

## 🎉 Next Steps

1. Train on full dataset for better accuracy
2. Test with your own sign language videos
3. Integrate into your application
4. Customize Gemini prompts for specific domains
