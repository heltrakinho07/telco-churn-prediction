"""
roi.py — Análise de retorno (ROI) em meticais (MZN).

Um modelo só interessa a uma empresa se gerar valor. Aqui traduzimos as
previsões do modelo em METICAIS: quanto se poupa ao usar o modelo para
dirigir uma campanha de retenção, em vez de não fazer nada.

NOTA: os valores abaixo são PRESSUPOSTOS de negócio — ajusta-os à
realidade da tua operadora. Estão todos no topo, bem visíveis.
"""
import json

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from config import MODELS_DIR, RANDOM_STATE, ROOT, TARGET
from data_prep import get_clean_data

# ======================================================================
# PRESSUPOSTOS DE NEGÓCIO (em meticais / MZN) — AJUSTA À TUA REALIDADE
# ======================================================================
ARPU_MENSAL = 500          # Receita média por cliente por mês (MZN)
MESES_VALOR = 12           # Horizonte: valor de reter por 12 meses
VALOR_CLIENTE = ARPU_MENSAL * MESES_VALOR   # Valor de vida (simplificado)

CUSTO_CAMPANHA = 300       # Custo de contactar/incentivar 1 cliente (MZN)
TAXA_SUCESSO = 0.30        # % de clientes em risco que a campanha consegue reter
# ======================================================================


def carregar_modelo_e_teste():
    """Carrega o XGBoost treinado e recria o mesmo conjunto de teste."""
    modelo = joblib.load(MODELS_DIR / "gradient_boosting_churn.joblib")
    df = get_clean_data(save=False)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    return modelo, X_test, y_test


def analisar_roi(modelo, X_test, y_test):
    """
    Simula o cenário: usamos o modelo para escolher que clientes contactar.

    Lógica:
      - O modelo marca clientes como 'em risco'.
      - Contactamos esses (custo = CUSTO_CAMPANHA cada).
      - De cada cliente que IA MESMO sair e é contactado, retemos
        uma fração (TAXA_SUCESSO), poupando o VALOR_CLIENTE.
    """
    y_pred = modelo.predict(X_test)

    # Verdadeiros positivos: previu churn E o cliente ia mesmo sair
    tp = int(((y_pred == 1) & (y_test == 1)).sum())
    # Falsos positivos: previu churn mas o cliente ia ficar (campanha "gasta")
    fp = int(((y_pred == 1) & (y_test == 0)).sum())

    contactados = tp + fp
    custo_total = contactados * CUSTO_CAMPANHA

    clientes_retidos = tp * TAXA_SUCESSO
    valor_salvo = clientes_retidos * VALOR_CLIENTE

    lucro_liquido = valor_salvo - custo_total
    roi_pct = (lucro_liquido / custo_total * 100) if custo_total else 0

    # Cenário alternativo: não fazer nada -> perde-se todos os que saem
    churners_reais = int((y_test == 1).sum())
    perda_sem_modelo = churners_reais * VALOR_CLIENTE

    return {
        "pressupostos": {
            "ARPU_mensal_MZN": ARPU_MENSAL,
            "meses_valor": MESES_VALOR,
            "valor_cliente_MZN": VALOR_CLIENTE,
            "custo_campanha_MZN": CUSTO_CAMPANHA,
            "taxa_sucesso_retencao": TAXA_SUCESSO,
        },
        "clientes_contactados": contactados,
        "custo_total_MZN": round(custo_total),
        "clientes_retidos_estimados": round(clientes_retidos, 1),
        "valor_salvo_MZN": round(valor_salvo),
        "lucro_liquido_MZN": round(lucro_liquido),
        "roi_percent": round(roi_pct, 1),
        "perda_potencial_sem_modelo_MZN": round(perda_sem_modelo),
    }


def imprimir(r):
    print("=" * 55)
    print("ANÁLISE DE ROI EM METICAIS (conjunto de teste)")
    print("=" * 55)
    p = r["pressupostos"]
    print(f"Pressupostos: ARPU {p['ARPU_mensal_MZN']} MZN/mês, "
          f"valor/cliente {p['valor_cliente_MZN']} MZN,")
    print(f"              custo campanha {p['custo_campanha_MZN']} MZN, "
          f"sucesso {p['taxa_sucesso_retencao']:.0%}")
    print("-" * 55)
    print(f"Clientes contactados:        {r['clientes_contactados']}")
    print(f"Custo da campanha:           {r['custo_total_MZN']:,} MZN")
    print(f"Clientes retidos (est.):     {r['clientes_retidos_estimados']}")
    print(f"Valor salvo:                 {r['valor_salvo_MZN']:,} MZN")
    print("-" * 55)
    print(f"LUCRO LÍQUIDO:               {r['lucro_liquido_MZN']:,} MZN")
    print(f"ROI:                         {r['roi_percent']}%")
    print(f"(Perda potencial sem modelo: {r['perda_potencial_sem_modelo_MZN']:,} MZN)")


def main():
    modelo, X_test, y_test = carregar_modelo_e_teste()
    r = analisar_roi(modelo, X_test, y_test)
    imprimir(r)

    out = ROOT / "reports" / "roi_phase2.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(r, f, indent=2, ensure_ascii=False)
    print(f"\nResultado guardado em: {out}")


if __name__ == "__main__":
    main()
