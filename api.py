import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from PIL import Image
import io
import urllib.request
import os

# Scarica il modello solo se non è già presente
MODEL_PATH = "food_model.pth"
MODEL_URL = "https://storage.googleapis.com/food-model-buckett/food_model.pth"

if not os.path.exists(MODEL_PATH):
    print("Scaricamento modello da Google Cloud...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Modello scaricato!")

# Crea il modello
class FoodModel(nn.Module):
    def __init__(self, num_classes=7):
        super(FoodModel, self).__init__()
        self.resnet = models.resnet18(weights='IMAGENET1K_V1')
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

model = FoodModel(num_classes=7)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# Preprocessing immagine
def get_transform():
    return transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# API FastAPI
app = FastAPI()

class PredictionResponse(BaseModel):
    prediction: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    transform = get_transform()
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
    _, predicted = torch.max(outputs, 1)

    class_names = ['apple_pie', 'cannoli', 'edamame', 'falafel', 'ramen', 'sushi', 'tiramisù']
    predicted_label = class_names[predicted.item()]

    return PredictionResponse(prediction=predicted_label)
