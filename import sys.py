import torch
import torch.nn as nn  # Aggiungi questa riga per importare nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

import os
model_file = 'food_model.pth'
if os.path.exists(model_file):
    os.remove(model_file)
    print("Vecchio modello eliminato!")
else:
    print("Il file del modello non esiste.")
