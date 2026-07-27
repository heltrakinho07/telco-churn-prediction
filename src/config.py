"""
config.py — Configuração central do projeto.

Ter os caminhos e constantes num só sítio é uma boa prática:
se algo mudar, muda-se aqui e não em 10 ficheiros diferentes.
"""
from pathlib import Path

# Raiz do projeto (a pasta acima de src/)
ROOT = Path(__file__).resolve().parent.parent

# Caminhos de dados
DATA_RAW = ROOT / "data" / "raw" / "telco_churn.csv"
DATA_PROCESSED = ROOT / "data" / "processed" / "telco_clean.csv"

# Caminhos de saída
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"

# Coluna que queremos prever (o "alvo" / target)
TARGET = "Churn"

# Semente aleatória: garante que os resultados são reproduzíveis
# (quem correr o teu código obtém exatamente os mesmos números)
RANDOM_STATE = 42
