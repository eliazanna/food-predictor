import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import os

# Definizione della classe FoodModel
class FoodModel(nn.Module):
    def __init__(self, num_classes=7):  # Modifica il numero di classi a 7
        super(FoodModel, self).__init__()
        self.resnet = models.resnet18(weights='IMAGENET1K_V1')  # Carica ResNet18 pre-addestrato
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)  # Modifica l'output per il tuo dataset

    def forward(self, x):
        return self.resnet(x)

# Funzione di preprocessing per le immagini
def get_transform():
    return transforms.Compose([
        transforms.Resize(224),  # Ridimensiona a 224x224
        transforms.CenterCrop(224),  # Ritaglia al centro
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizzazione
    ])

# Funzione di allenamento
def train(model, train_loader, optimizer, criterion, device, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoca {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}')

# Caricamento o allenamento del modello
def load_or_train_model():
    model = FoodModel(num_classes=7)  # Assicurati di creare l'istanza del modello
    model_file = 'food_model.pth'

    # Se il modello è già stato salvato, non eseguire l'allenamento
    if os.path.exists(model_file):
        print(f"Caricamento del modello esistente da {model_file}...")
        model.load_state_dict(torch.load(model_file))  # Carica i pesi salvati
        model.eval()  # Imposta il modello in modalità di previsione
        print("Modello caricato con successo!")
    else:
        print("Allenamento del modello...")
        # Creazione del dataset e del dataloader
        data_dir = 'C:/Users/eliza/Desktop/calorie-predictor/food-101-tiny'
        train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=get_transform())
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Creazione del modello, ottimizzatore e funzione di perdita
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Allenamento del modello
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        train(model, train_loader, optimizer, criterion, device, epochs=5)

        # Salvataggio del modello
        torch.save(model.state_dict(), model_file)
        print(f"Modello allenato e salvato come {model_file}.")
    
    return model

#carico/alleno il modello
model = load_or_train_model()

