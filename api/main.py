"""
main.py — API REST de previsão de churn (FastAPI).

Expõe o modelo através de HTTP. Qualquer sistema (site, CRM, app) pode
enviar os dados de um cliente e receber a previsão em JSON.

Correr localmente:
    uvicorn api.main:app --reload
Depois abrir http://localhost:8000/docs para a documentação interativa.
"""
import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

# Permitir importar o módulo partilhado em src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from predict import prever  # noqa: E402

app = FastAPI(
    title="API de Previsão de Churn — Telco",
    description="Prevê se um cliente de telecomunicações vai cancelar o serviço.",
    version="1.0.0",
)


class Cliente(BaseModel):
    """Dados de entrada de um cliente. Os exemplos aparecem no /docs."""
    gender: str = Field(..., example="Female")
    SeniorCitizen: str = Field(..., example="No")
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., ge=0, example=12)
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="Fiber optic")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="No")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="Yes")
    StreamingMovies: str = Field(..., example="Yes")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., ge=0, example=85.0)
    TotalCharges: float = Field(..., ge=0, example=1000.0)


class Previsao(BaseModel):
    churn: bool
    probabilidade: float
    risco: str


@app.get("/")
def raiz():
    """Verificação rápida de que a API está viva."""
    return {"mensagem": "API de previsão de churn ativa. Ver /docs."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Previsao)
def predict(cliente: Cliente):
    """Recebe os dados de um cliente e devolve a previsão de churn."""
    return prever(cliente.model_dump())
