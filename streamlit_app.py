import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Modello custom
class FoodModel(nn.Module):
    def __init__(self, num_classes=7):
        super(FoodModel, self).__init__()
        from torchvision import models
        self.resnet = models.resnet18(weights='IMAGENET1K_V1')
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Trasformazione
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Caricamento modello
@st.cache_resource
def load_model():
    model = FoodModel()
    model.load_state_dict(torch.load("food_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# Interfaccia Streamlit
st.title("🍕 Calorie Predictor - Upload immagine cibo")

uploaded_file = st.file_uploader("Carica un'immagine", type=["jpg", "png", "jpeg"])
class_names = ['apple_pie', 'cannoli', 'edamame', 'falafel', 'ramen', 'sushi', 'tiramisù']

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Immagine caricata", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        predicted_label = class_names[predicted.item()]
    
    st.success(f"🍽 Questo cibo è: **{predicted_label}**")
