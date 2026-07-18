import os
import torch
import numpy as np
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "./models/bioclinicalbert_pipeline"
ENCODER_PATH = "./models/transformer_label_encoder.pkl"

app = FastAPI(
    title="Omega Clinical Route Core Engine",
    description="Automated Transformer pipeline for hospital intake workflows."
)

# 1. Initialize mapping elements and model paths
MODEL_DIR = "./models/bioclinicalbert_pipeline"
ENCODER_PATH = "./models/transformer_label_encoder.pkl"

if not os.path.exists(MODEL_DIR) or not os.path.exists(ENCODER_PATH):
    raise RuntimeError("Missing transformer weight arrays! Run your training steps first.")

# Load fine-tuned assets into memory
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

# 2. Strict type verification payloads
class IntakeTranscript(BaseModel):
    note: str = Field(..., min_length=15, example="Patient shows sign of acute chest pain.")

@app.post("/api/v1/route")
async def route_clinical_note(payload: IntakeTranscript):
    try:
        # Tokenize incoming raw textual notes
        inputs = tokenizer(payload.note, return_tensors="pt", truncation=True, padding=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Inference calculations
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
        predicted_class_id = int(np.argmax(probabilities))
        confidence_score = float(probabilities[predicted_class_id])
        assigned_specialty = label_encoder.inverse_transform([predicted_class_id])[0]
        
        return {
            "status": "success",
            "routing_metrics": {
                "assigned_specialty": assigned_specialty.strip(),
                "confidence_score": round(confidence_score, 3),
                "engine": "Fine-Tuned BioClinicalBERT Transformer"
            }
        }
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Inference crash: {str(err)}")
