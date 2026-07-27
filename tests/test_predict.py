"""
Testes da lógica de previsão (predict.prever).

Testa o modelo REAL guardado, de ponta a ponta: dá um cliente, verifica
que a previsão tem o formato certo e faz sentido de negócio.
"""
import pytest

from predict import (
    EXEMPLO_CLIENTE,
    FEATURE_ORDER,
    analisar_parametros,
    prever,
    prever_lote,
)


def test_formato_da_resposta():
    """A previsão deve ter as chaves esperadas e tipos corretos."""
    r = prever(EXEMPLO_CLIENTE)
    assert set(r.keys()) == {"churn", "probabilidade", "risco"}
    assert isinstance(r["churn"], bool)
    assert isinstance(r["probabilidade"], float)
    assert r["risco"] in {"Baixo", "Médio", "Alto"}


def test_probabilidade_entre_0_e_1():
    """A probabilidade tem de ser uma probabilidade válida."""
    r = prever(EXEMPLO_CLIENTE)
    assert 0.0 <= r["probabilidade"] <= 1.0


def test_cliente_de_risco_vs_fiel():
    """
    Teste de sentido de negócio: um cliente mensal, novo e sem serviços
    de apoio deve ter MAIOR probabilidade de churn do que um cliente
    com contrato de 2 anos, antigo e com todos os serviços.
    """
    risco = dict(EXEMPLO_CLIENTE)
    risco.update(Contract="Month-to-month", tenure=1, TechSupport="No",
                 OnlineSecurity="No")

    fiel = dict(EXEMPLO_CLIENTE)
    fiel.update(Contract="Two year", tenure=70, TechSupport="Yes",
                OnlineSecurity="Yes", InternetService="DSL")

    p_risco = prever(risco)["probabilidade"]
    p_fiel = prever(fiel)["probabilidade"]
    assert p_risco > p_fiel


def test_consistencia_mesma_entrada():
    """A mesma entrada deve dar sempre a mesma saída (determinístico)."""
    r1 = prever(EXEMPLO_CLIENTE)
    r2 = prever(EXEMPLO_CLIENTE)
    assert r1 == r2


def test_prever_lote_igual_a_prever_individual():
    """Prever em lote tem de dar o mesmo que prever um a um."""
    fiel = dict(EXEMPLO_CLIENTE, Contract="Two year", tenure=70)
    probs = prever_lote([EXEMPLO_CLIENTE, fiel])
    assert len(probs) == 2
    assert probs[0] == pytest.approx(prever(EXEMPLO_CLIENTE)["probabilidade"], abs=1e-4)
    assert probs[1] == pytest.approx(prever(fiel)["probabilidade"], abs=1e-4)


def test_analise_cobre_todos_os_parametros():
    """A previsão completa tem de avaliar as 19 variáveis do modelo."""
    analise = analisar_parametros(EXEMPLO_CLIENTE)
    avaliados = {p["parametro"] for p in analise["parametros"]}
    assert avaliados == set(FEATURE_ORDER)
    assert analise["probabilidade"] == prever(EXEMPLO_CLIENTE)["probabilidade"]


def test_analise_ordenada_e_coerente():
    """Parâmetros ordenados por influência; melhor cenário nunca é pior que o atual."""
    analise = analisar_parametros(EXEMPLO_CLIENTE)
    influencias = [p["influencia"] for p in analise["parametros"]]
    assert influencias == sorted(influencias, reverse=True)

    for p in analise["parametros"]:
        assert p["influencia"] >= 0
        assert p["melhor_prob"] <= analise["probabilidade"] + 1e-9
        assert p["pior_prob"] >= analise["probabilidade"] - 1e-9
