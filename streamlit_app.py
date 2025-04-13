import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import uuid
import csv
from datetime import datetime
import os

st.set_page_config(page_title="Food Classifier", layout="centered")

# ----------------- Funzioni di supporto -----------------

def get_or_create_user_id():
    if 'user_id' not in st.session_state:
        st.session_state['user_id'] = str(uuid.uuid4())
    return st.session_state['user_id']

def log_user_image(user_id, filename, prediction, log_path='user_tracking.csv'):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.exists(log_path)

    with open(log_path, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['user_id', 'filename', 'prediction', 'timestamp'])
        writer.writerow([user_id, filename, prediction, now])

# ----------------- Modello -----------------

class FoodModel(nn.Module):
    def __init__(self, num_classes=7):
        super(FoodModel, self).__init__()
        from torchvision import models
        self.resnet = models.resnet18(weights='IMAGENET1K_V1')
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

@st.cache_resource
def load_model():
    model = FoodModel()
    model.load_state_dict(torch.load("food_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

# ----------------- Preprocessing -----------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

model = load_model()

# ----------------- UI Streamlit -----------------

st.title("🍽️ Calorie Predictor")
st.caption("Scatta o carica una foto del tuo piatto per scoprire cosa stai mangiando!")

uploaded_file = st.file_uploader("Carica un'immagine", type=["jpg", "png", "jpeg"])
class_names = ['apple_pie', 'cannoli', 'edamame', 'falafel', 'ramen', 'sushi', 'tiramisù']

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Immagine caricata", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        predicted_label = class_names[predicted.item()]

    st.success(f"🍝 Questo cibo sembra: **{predicted_label.upper()}**")

    # Tracking dell'utente
    user_id = get_or_create_user_id()
    log_user_image(user_id, uploaded_file.name, predicted_label)
