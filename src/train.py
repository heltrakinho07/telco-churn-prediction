"""
train.py — Treino e avaliação do modelo baseline.

"Baseline" = um primeiro modelo simples que serve de referência.
Se um modelo mais complexo não bater o baseline, não vale a complexidade.
Aqui usamos Regressão Logística — simples, rápida e interpretável.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import MODELS_DIR, RANDOM_STATE, TARGET
from data_prep import get_clean_data


def build_pipeline(numeric_cols, categorical_cols) -> Pipeline:
    """
    Constrói um Pipeline que faz DUAS coisas em cadeia:
      1) Pré-processamento (escalar números, codificar categorias)
      2) O modelo (Regressão Logística)

    Porquê Pipeline? Porque garante que o pré-processamento é "aprendido"
    APENAS com os dados de treino e depois aplicado ao teste — evitando
    o erro clássico de 'data leakage' (contaminar o teste com info do treino).
    """
    preprocessor = ColumnTransformer(
        transformers=[
            # Números: colocar na mesma escala
            ("num", StandardScaler(), numeric_cols),
            # Categorias: transformar em colunas 0/1 (one-hot)
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    model = LogisticRegression(
        max_iter=1000,
        # 'balanced' dá mais peso à classe minoritária (quem faz churn).
        # Sem isto, o modelo tende a "prever sempre que fica" — inútil.
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    return Pipeline(steps=[("prep", preprocessor), ("model", model)])


def main():
    # 1) Carregar dados limpos
    df = get_clean_data(save=False)

    # 2) Separar variáveis (X) do alvo (y)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Identificar automaticamente colunas numéricas vs. categóricas
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    # 3) Dividir em treino (80%) e teste (20%).
    #    'stratify=y' mantém a mesma proporção de churn nos dois conjuntos.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # 4) Construir e treinar
    pipe = build_pipeline(numeric_cols, categorical_cols)
    pipe.fit(X_train, y_train)

    # 5) Avaliar no conjunto de TESTE (dados que o modelo nunca viu)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    print("=" * 55)
    print("RESULTADOS DO MODELO BASELINE (Regressão Logística)")
    print("=" * 55)
    print("\nRelatório de classificação:")
    print(classification_report(y_test, y_pred,
                                target_names=["Ficou (0)", "Saiu (1)"]))

    print("Matriz de confusão:")
    print(confusion_matrix(y_test, y_pred))
    print("  (linhas = real, colunas = previsto)")

    auc = roc_auc_score(y_test, y_proba)
    print(f"\nROC-AUC: {auc:.3f}")
    print("  (0.5 = aleatório | 1.0 = perfeito | >0.80 = bom)")

    # 6) Guardar o modelo treinado para usar depois (deploy)
    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / "baseline_logreg.joblib"
    joblib.dump(pipe, model_path)
    print(f"\nModelo guardado em: {model_path}")


if __name__ == "__main__":
    main()
