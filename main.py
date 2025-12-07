from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from PIL import Image
import joblib
import numpy as np
import io
import sys, os

sys.path.append("..")
from feature_extractor import extract_features

app = FastAPI()

models_path = "C:/Users/ADMIN/animal-classifier/src/api/models"

# Load all models
models = {
    "logreg": joblib.load(f"{models_path}/logreg.joblib"),
    "knn": joblib.load(f"{models_path}/knn.joblib"),
    "gnb": joblib.load(f"{models_path}/gnb.joblib"),
    "linsvc": joblib.load(f"{models_path}/linsvc.joblib"),
}

le = joblib.load(f"{models_path}/label_encoder.joblib")

# Serve frontend HTML
@app.get("/", response_class=HTMLResponse)
def home():
    return FileResponse("static/index.html")  # ✅ Safer than open()

# Predict endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    feat = extract_features(img).reshape(1, -1)

    results = {}
    for name, model in models.items():
        pred = model.predict(feat)[0]
        label = le.inverse_transform([pred])[0]
        results[name] = label

    return results
