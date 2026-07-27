"""
predict.py — Lógica partilhada de previsão.

A API (FastAPI) e a interface (Streamlit) usam ESTE módulo, para garantir
que ambas fazem exatamente a mesma coisa. Um só sítio para a verdade.
"""
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "gradient_boosting_churn.joblib"

# Ordem exata das colunas que o modelo espera
FEATURE_ORDER = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]

# Valores possíveis de cada variável categórica (para menus e validação)
CATEGORICAL_OPTIONS = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": ["No", "Yes"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "No internet service", "Yes"],
    "OnlineBackup": ["No", "No internet service", "Yes"],
    "DeviceProtection": ["No", "No internet service", "Yes"],
    "TechSupport": ["No", "No internet service", "Yes"],
    "StreamingTV": ["No", "No internet service", "Yes"],
    "StreamingMovies": ["No", "No internet service", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Bank transfer (automatic)", "Credit card (automatic)",
        "Electronic check", "Mailed check",
    ],
}

# Um exemplo típico (cliente de risco médio) — útil para pré-preencher formulários
EXEMPLO_CLIENTE = {
    "gender": "Female", "SeniorCitizen": "No", "Partner": "Yes",
    "Dependents": "No", "tenure": 12, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
    "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "Yes",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 85.0,
    "TotalCharges": 1000.0,
}

# Valores alternativos testados nas variáveis numéricas durante a análise de
# sensibilidade (cobrem a gama observada no dataset original).
NUMERIC_GRID = {
    "tenure": [0, 6, 12, 24, 36, 48, 60, 72],
    "MonthlyCharges": [20.0, 35.0, 50.0, 70.0, 90.0, 110.0],
    "TotalCharges": [100.0, 500.0, 1500.0, 3000.0, 5000.0, 8000.0],
}

# Nomes legíveis (em português) de cada parâmetro, para relatórios e interface
NOMES_PARAMETROS = {
    "gender": "Género",
    "SeniorCitizen": "Idoso (65+)",
    "Partner": "Parceiro/a",
    "Dependents": "Dependentes",
    "tenure": "Tempo de casa (meses)",
    "PhoneService": "Serviço telefónico",
    "MultipleLines": "Múltiplas linhas",
    "InternetService": "Serviço de internet",
    "OnlineSecurity": "Segurança online",
    "OnlineBackup": "Backup online",
    "DeviceProtection": "Proteção de equipamento",
    "TechSupport": "Suporte técnico",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming filmes",
    "Contract": "Tipo de contrato",
    "PaperlessBilling": "Fatura eletrónica",
    "PaymentMethod": "Método de pagamento",
    "MonthlyCharges": "Encargo mensal (MZN)",
    "TotalCharges": "Total gasto (MZN)",
}

_modelo = None


def carregar_modelo():
    """Carrega o modelo uma só vez (cache) e reutiliza."""
    global _modelo
    if _modelo is None:
        _modelo = joblib.load(MODEL_PATH)
    return _modelo


def classificar_risco(proba: float) -> str:
    """Traduz uma probabilidade (0 a 1) na etiqueta de risco usada em todo o projeto."""
    if proba < 0.35:
        return "Baixo"
    if proba < 0.65:
        return "Médio"
    return "Alto"


def prever_lote(lista_dados: list[dict]) -> list[float]:
    """
    Prevê vários clientes de uma só vez e devolve as probabilidades de churn.

    Uma única chamada ao modelo é muito mais rápida do que N chamadas
    individuais — útil para a análise de sensibilidade de todos os parâmetros.
    """
    modelo = carregar_modelo()
    linhas = [{col: dados[col] for col in FEATURE_ORDER} for dados in lista_dados]
    df = pd.DataFrame(linhas, columns=FEATURE_ORDER)
    return [float(p) for p in modelo.predict_proba(df)[:, 1]]


def alternativas_parametro(parametro: str, valor_atual) -> list:
    """Valores alternativos a testar num parâmetro (exclui o valor atual do cliente)."""
    if parametro in CATEGORICAL_OPTIONS:
        candidatos = CATEGORICAL_OPTIONS[parametro]
    else:
        candidatos = NUMERIC_GRID[parametro]
    return [v for v in candidatos if v != valor_atual]


def analisar_parametros(dados: dict) -> dict:
    """
    Previsão completa: usa TODOS os parâmetros e mede o peso de cada um.

    Para cada uma das 19 variáveis, mantém o resto do perfil fixo e volta a
    prever com os valores alternativos dessa variável. A diferença face à
    previsão original mostra quanto esse parâmetro pesa neste cliente concreto
    e qual o valor que mais reduziria o risco (alavanca de retenção).

    Retorna:
      - probabilidade / churn / risco: a previsão do cliente tal como está
      - parametros: uma lista (ordenada por influência) com, por parâmetro,
        o valor atual, o melhor e o pior cenário e a amplitude de influência
    """
    base = prever(dados)
    prob_base = base["probabilidade"]

    # Montar todos os cenários de uma vez e prever num só lote
    cenarios, indice = [], []
    for parametro in FEATURE_ORDER:
        for alternativa in alternativas_parametro(parametro, dados[parametro]):
            cenarios.append({**dados, parametro: alternativa})
            indice.append((parametro, alternativa))

    probs = prever_lote(cenarios) if cenarios else []

    # Agrupar os resultados por parâmetro
    por_parametro = {parametro: [] for parametro in FEATURE_ORDER}
    for (parametro, alternativa), prob in zip(indice, probs):
        por_parametro[parametro].append((alternativa, prob))

    linhas = []
    for parametro in FEATURE_ORDER:
        testes = por_parametro[parametro] + [(dados[parametro], prob_base)]
        melhor_valor, melhor_prob = min(testes, key=lambda t: t[1])
        pior_valor, pior_prob = max(testes, key=lambda t: t[1])
        linhas.append({
            "parametro": parametro,
            "nome": NOMES_PARAMETROS[parametro],
            "valor_atual": dados[parametro],
            "influencia": round(pior_prob - melhor_prob, 4),
            "melhor_valor": melhor_valor,
            "melhor_prob": round(melhor_prob, 4),
            "ganho": round(prob_base - melhor_prob, 4),
            "pior_valor": pior_valor,
            "pior_prob": round(pior_prob, 4),
        })

    linhas.sort(key=lambda linha: linha["influencia"], reverse=True)
    return {**base, "parametros": linhas}


def prever(dados: dict) -> dict:
    """
    Recebe os dados de um cliente (dict) e devolve a previsão.

    Retorna:
      - churn: True/False (vai sair?)
      - probabilidade: chance de churn (0 a 1)
      - risco: etiqueta legível (Baixo / Médio / Alto)
    """
    modelo = carregar_modelo()

    # Construir uma linha na ordem certa das colunas
    linha = {col: dados[col] for col in FEATURE_ORDER}
    df = pd.DataFrame([linha], columns=FEATURE_ORDER)

    proba = float(modelo.predict_proba(df)[0, 1])
    churn = bool(proba >= 0.5)

    return {
        "churn": churn,
        "probabilidade": round(proba, 4),
        "risco": classificar_risco(proba),
    }
