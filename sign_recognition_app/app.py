from fastapi import FastAPI, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import os
import json
from inference.predict import SignLanguagePredictor
from inference.gemini_api import generate_sentence_from_signs

app = FastAPI()

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load LSTM model (loads once when app starts)
predictor = None

@app.on_event("startup")
async def startup_event():
    global predictor
    try:
        predictor = SignLanguagePredictor(
            model_path="inference/lstm_model_best.keras",
            label_map_path="inference/label_map.json"
        )
        print("✅ LSTM model loaded successfully!")
    except Exception as e:
        print(f"⚠️  Warning: Could not load model: {e}")
        print("   The app will start but predictions won't work until you train a model.")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict_single(video: UploadFile):
    """
    Predict a SINGLE sign from uploaded video.
    Returns predicted sign with confidence.
    """
    if predictor is None:
        return JSONResponse(
            status_code=500,
            content={"error": "Model not loaded. Please train a model first."}
        )
    
    # Save uploaded video
    uploads = Path("uploads")
    uploads.mkdir(exist_ok=True)
    video_path = uploads / video.filename
    
    with open(video_path, "wb") as f:
        f.write(await video.read())
    
    try:
        # Get prediction
        result = predictor.predict_sign(str(video_path))
        
        # Get top 5 predictions for additional context
        top_predictions = predictor.predict_top_k(str(video_path), k=5)
        
        return {
            "mode": "single",
            "predicted_label": result["label_id"],
            "predicted_sign": result["sign_text"],
            "confidence": result["confidence"],
            "sentence": result["sign_text"],  # Single sign = the sign itself
            "top_predictions": top_predictions
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Prediction failed: {str(e)}"}
        )

@app.post("/predict-sequence")
async def predict_sequence(video: UploadFile, method: str = Form("motion")):
    """
    Predict MULTIPLE signs from uploaded video and generate a sentence.
    
    Args:
        video: Uploaded video file
        method: Segmentation method - "motion" or "fixed"
    
    Returns:
        JSON with sign sequence and generated sentence
    """
    if predictor is None:
        return JSONResponse(
            status_code=500,
            content={"error": "Model not loaded. Please train a model first."}
        )
    
    # Save uploaded video
    uploads = Path("uploads")
    uploads.mkdir(exist_ok=True)
    video_path = uploads / video.filename
    
    with open(video_path, "wb") as f:
        f.write(await video.read())
    
    try:
        # Get sequence prediction
        result = predictor.predict_sequence(
            str(video_path),
            method=method,
            confidence_threshold=0.3
        )
        
        # Generate sentence from sign sequence using Gemini
        sign_sequence = result["sign_sequence"]
        
        if len(sign_sequence) > 0:
            sentence = generate_sentence_from_signs(sign_sequence)
        else:
            sentence = "[No signs detected]"
        
        return {
            "mode": "sequence",
            "sign_sequence": sign_sequence,
            "signs": result["signs"],
            "sentence": sentence,
            "num_segments": result["num_segments"],
            "num_signs_detected": result["num_signs_detected"]
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Sequence prediction failed: {str(e)}"}
        )

@app.get("/model-info")
async def model_info():
    """Get information about the loaded model."""
    if predictor is None:
        return {"loaded": False, "message": "No model loaded"}
    
    return {
        "loaded": True,
        "num_classes": len(predictor.label_map),
        "classes": list(predictor.label_map.values())[:10]  # Show first 10
    }


