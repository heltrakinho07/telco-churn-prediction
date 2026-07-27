# Dockerfile para deploy no Hugging Face Spaces (interface Streamlit).
# O HF Spaces expõe a porta 7860.
FROM python:3.12-slim

WORKDIR /app

# Instalar dependências primeiro (aproveita a cache do Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o código e o modelo treinado
COPY src/ ./src/
COPY app/ ./app/
COPY models/gradient_boosting_churn.joblib ./models/gradient_boosting_churn.joblib

EXPOSE 7860

# Arrancar a interface Streamlit na porta do HF Spaces
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.headless=true"]
