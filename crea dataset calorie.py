import csv

# Lista di tuple (cibo, calorie)
dati = [
    ("apple_pie", 296),
    ("cannoli", 216),
    ("edamame", 122),
    ("falafel", 333),
    ("ramen", 436),
    ("sushi", 200),
    ("tiramisù", 358)
]

# Salva nel file CSV senza intestazioni
with open("calorie.csv", mode="w", newline='', encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerows(dati)

print("✅ File calorie.csv creato con successo.")
