import joblib
import os

MODEL_PATH = os.path.join("inference", "rf_model.pkl")
model = joblib.load(MODEL_PATH)
