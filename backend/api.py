from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
from torchvision import transforms
import io
import os

# For X-ray model (Hugging Face Transformers)
from transformers import AutoImageProcessor, AutoModelForImageClassification

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load X-ray model and processor
X_RAY_MODEL_PATH = "./backend/saved_xray_model"
xray_processor = AutoImageProcessor.from_pretrained(X_RAY_MODEL_PATH)
xray_model = AutoModelForImageClassification.from_pretrained(X_RAY_MODEL_PATH)
xray_model = xray_model.to('cpu') 


SKIN_MODEL_PATH = "./backend/saved_skin_model/skinconvnext_scripted.pt"
skin_model = torch.jit.load(SKIN_MODEL_PATH, map_location=torch.device('cpu'))  
skin_model.eval()

# Define skin class labels for Eraly-ml/Skin-AI model
skin_class_labels = [
    "Acne and Rosacea",
    "Actinic Keratosis Basal Cell Carcinoma",
    "Atopic Dermatitis",
    "Bullous Disease",
    "Cellulitis Impetigo",
    "Eczema",
    "Exanthems and Drug Eruptions",
    "Hair Loss Photos Alopecia and other Hair Diseases",
    "Herpes HPV and other STDs",
    "Light Diseases and Disorders of Pigmentation",
    "Lupus and other Connective Tissue diseases",
    "Melanoma Skin Cancer Nevi and Moles",
    "Nail Fungus and other Nail Disease",
    "Poison Ivy Photos and other Contact Dermatitis",
    "Psoriasis pictures Lichen Planus and related diseases",
    "Scabies Lyme Disease and other Infestations and Bites",
    "Seborrheic Keratoses and other Benign Tumors",
    "Systemic Disease",
    "Tinea Ringworm Candidiasis and other Fungal Infections",
    "Urticaria Hives",
    "Vascular Tumors",
    "Vasculitis Photos",
    "Warts Molluscum and other Viral Infections"
]


xray_class_labels = [
    'Cardiomegaly', 'Edema', 'Consolidation', 'Pneumonia', 'No Finding'
]


skin_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.post("/diagnose-xray")
async def diagnose_xray(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = xray_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = xray_model(**inputs)
        logits = outputs.logits
        predicted_class_idx = torch.argmax(logits, dim=-1).item()
        predicted_label = xray_class_labels[predicted_class_idx]
        return {"prediction": predicted_label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"X-ray diagnosis failed: {str(e)}")

@app.post("/diagnose-skin")
async def diagnose_skin(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = skin_transform(image).unsqueeze(0)
        with torch.no_grad():
            output = skin_model(image_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
       
        if predicted_class < len(skin_class_labels):
            predicted_label = skin_class_labels[predicted_class]
        else:
            predicted_label = str(predicted_class)
        return {"prediction": predicted_label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Skin diagnosis failed: {str(e)}")

@app.get("/")
def root():
    return {"message": "Medical Diagnosis API is running."} 