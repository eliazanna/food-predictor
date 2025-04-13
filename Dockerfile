FROM python:3.10-slim

# Imposta la directory di lavoro
WORKDIR /app

# Copia i file necessari
COPY . .

# Installa le dipendenze
RUN pip install --no-cache-dir -r requirements.txt

# Espone la porta usata da Streamlit
EXPOSE 8501

# Comando per avviare l'app
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.enableCORS=false"]
