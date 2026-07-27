"""
Testes da limpeza de dados (data_prep.clean).

Usamos um mini-dataset criado à mão, com os casos-problema conhecidos,
para não depender do ficheiro de dados completo.
"""
import pandas as pd

from data_prep import clean


def _mini_df():
    """Cria 2 clientes de exemplo, incluindo o caso do TotalCharges em branco."""
    return pd.DataFrame([
        {
            "customerID": "0001-AAA", "gender": "Female", "SeniorCitizen": 0,
            "Partner": "Yes", "Dependents": "No", "tenure": 5,
            "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "DSL", "OnlineSecurity": "Yes",
            "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "Yes",
            "StreamingTV": "No", "StreamingMovies": "No", "Contract": "One year",
            "PaperlessBilling": "No", "PaymentMethod": "Mailed check",
            "MonthlyCharges": 55.0, "TotalCharges": "275.0", "Churn": "No",
        },
        {
            "customerID": "0002-BBB", "gender": "Male", "SeniorCitizen": 1,
            "Partner": "No", "Dependents": "No", "tenure": 0,
            "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "Fiber optic", "OnlineSecurity": "No",
            "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
            "StreamingTV": "Yes", "StreamingMovies": "Yes",
            "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check", "MonthlyCharges": 90.0,
            "TotalCharges": " ", "Churn": "Yes",  # branco (cliente novo)
        },
    ])


def test_remove_customer_id():
    """A coluna customerID não deve sobrar (não ajuda a prever)."""
    out = clean(_mini_df())
    assert "customerID" not in out.columns


def test_totalcharges_vira_numero():
    """TotalCharges deve ficar numérica e o branco vira 0."""
    out = clean(_mini_df())
    assert pd.api.types.is_numeric_dtype(out["TotalCharges"])
    # O cliente novo (tenure=0) tinha TotalCharges em branco -> 0
    assert out.loc[out["tenure"] == 0, "TotalCharges"].iloc[0] == 0


def test_churn_vira_binario():
    """Churn deve ser convertido para 0/1."""
    out = clean(_mini_df())
    assert set(out["Churn"].unique()).issubset({0, 1})
    assert out["Churn"].tolist() == [0, 1]  # No -> 0, Yes -> 1


def test_senior_citizen_legivel():
    """SeniorCitizen (0/1) deve virar 'No'/'Yes'."""
    out = clean(_mini_df())
    assert set(out["SeniorCitizen"].unique()).issubset({"No", "Yes"})


def test_sem_valores_em_falta():
    """Depois da limpeza não deve haver NaN."""
    out = clean(_mini_df())
    assert out.isna().sum().sum() == 0
