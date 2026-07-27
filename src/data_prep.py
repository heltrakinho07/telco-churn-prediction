"""
data_prep.py — Carregamento e limpeza dos dados.

Objetivo: transformar o CSV bruto num dataset limpo e pronto a modelar.
Isto é ~80% do trabalho real de um cientista de dados.
"""
import pandas as pd

from config import DATA_RAW, DATA_PROCESSED, TARGET


def load_raw() -> pd.DataFrame:
    """Lê o CSV bruto tal como veio da fonte."""
    return pd.read_csv(DATA_RAW)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Limpa o dataset e devolve uma versão pronta a usar."""
    df = df.copy()

    # 1) A coluna 'customerID' é só um identificador — não ajuda a prever nada.
    df = df.drop(columns=["customerID"])

    # 2) PROBLEMA CONHECIDO deste dataset:
    #    'TotalCharges' foi lida como texto porque tem 11 valores em branco
    #    (clientes com tenure=0, ou seja, acabaram de entrar).
    #    Convertemos para número; os brancos viram NaN (valor em falta).
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    #    Esses 11 clientes têm tenure=0, logo faz sentido TotalCharges=0.
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # 3) 'SeniorCitizen' já vem como 0/1. Vamos torná-la legível (No/Yes)
    #    para ser tratada como as outras variáveis categóricas.
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # 4) O alvo 'Churn' vem como 'Yes'/'No'. Modelos precisam de números:
    #    Yes (cliente saiu) -> 1  |  No (ficou) -> 0
    df[TARGET] = df[TARGET].map({"No": 0, "Yes": 1})

    return df


def get_clean_data(save: bool = True) -> pd.DataFrame:
    """Função principal: carrega, limpa e (opcionalmente) grava."""
    df = clean(load_raw())
    if save:
        df.to_csv(DATA_PROCESSED, index=False)
    return df


if __name__ == "__main__":
    df = get_clean_data()
    print(f"Dataset limpo: {df.shape[0]} clientes, {df.shape[1]} colunas")
    print(f"Taxa de churn: {df['Churn'].mean():.1%}")
    print(f"Guardado em: {DATA_PROCESSED}")
