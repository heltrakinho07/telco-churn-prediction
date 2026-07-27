"""
explain.py — Explicabilidade do modelo com SHAP.

Prever churn não chega: a empresa quer saber PORQUÊ. O SHAP mede o
contributo de cada variável para a previsão. Isto é essencial em setores
regulados (banca, telco) e mostra maturidade de Data Science.

Usamos um explicador model-agnostic (funciona com qualquer modelo), aplicado
a uma amostra do conjunto de teste para ser rápido.
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.model_selection import train_test_split

from config import FIGURES_DIR, MODELS_DIR, RANDOM_STATE, TARGET
from data_prep import get_clean_data


def main():
    modelo = joblib.load(MODELS_DIR / "gradient_boosting_churn.joblib")

    df = get_clean_data(save=False)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    _, X_test, _, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # O modelo é um Pipeline (prep + gb). Separamos as duas partes para
    # trabalhar com os dados já pré-processados e nomes de variáveis legíveis.
    preprocessor = modelo.named_steps["prep"]
    classifier = modelo.named_steps["model"]

    X_test_prep = preprocessor.transform(X_test)
    feature_names = list(preprocessor.get_feature_names_out())

    # Amostras: fundo (referência) e alvo (o que explicamos). Manter pequeno
    # torna o cálculo rápido sem perder qualidade da explicação.
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(X_test_prep.shape[0], size=min(200, X_test_prep.shape[0]),
                     replace=False)
    background = shap.sample(X_test_prep, 100, random_state=RANDOM_STATE)
    X_sample = X_test_prep[idx]

    print("A calcular valores SHAP (pode demorar ~1 min)...")
    # Probabilidade da classe 'churn' (índice 1)
    f = lambda data: classifier.predict_proba(data)[:, 1]
    explainer = shap.KernelExplainer(f, background)
    shap_values = explainer.shap_values(X_sample, nsamples=100)

    # Gráfico 1: importância global (barras)
    plt.figure()
    shap.summary_plot(
        shap_values, X_sample, feature_names=feature_names,
        plot_type="bar", show=False, max_display=12,
    )
    plt.title("Variáveis mais importantes para prever churn (SHAP)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_shap_importance.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Gráfico 2: beeswarm (direção do efeito de cada variável)
    plt.figure()
    shap.summary_plot(
        shap_values, X_sample, feature_names=feature_names,
        show=False, max_display=12,
    )
    plt.title("Efeito de cada variável no churn (SHAP)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "05_shap_beeswarm.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Top variáveis em texto (para o documento de resultados)
    importancia = np.abs(shap_values).mean(axis=0)
    ordem = np.argsort(importancia)[::-1][:10]
    print("\n--- Top 10 variáveis (importância SHAP média) ---")
    for i in ordem:
        print(f"  {feature_names[i]:40s}: {importancia[i]:.4f}")

    print(f"\n2 figuras SHAP guardadas em: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
