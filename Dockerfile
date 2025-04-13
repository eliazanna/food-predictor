FROM python:3.10-slim
WORKDIR /app
COPY . .

# Installa pacchetti manualmente
RUN pip install --no-cache-dir torch torchvision fastapi streamlit pillow

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8080", "--server.enableCORS=false"]
