"""
train_advanced.py — Modelo avançado (Gradient Boosting) com validação cruzada.

Fase 2. Dois saltos de qualidade:
  1) HistGradientBoosting — um modelo de boosting poderoso e rápido,
     do próprio scikit-learn (mesma família de ideias do XGBoost/LightGBM).
  2) Validação cruzada (cross-validation) — para PROVAR que o desempenho
     é CONSISTENTE e não sorte de uma divisão específica dos dados.

No fim, comparamos com o baseline e guardamos as métricas num JSON,
para alimentar o documento de resultados.
"""
import json

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import MODELS_DIR, RANDOM_STATE, ROOT, TARGET
from data_prep import get_clean_data


def build_preprocessor(numeric_cols, categorical_cols) -> ColumnTransformer:
    """Pré-processamento partilhado por todos os modelos."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )


def build_gb_pipeline(preprocessor) -> Pipeline:
    """
    Pipeline com HistGradientBoosting.

    'class_weight="balanced"' resolve o desequilíbrio de classes, dando mais
    peso à classe minoritária (os clientes que fazem churn).
    """
    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=4,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    return Pipeline(steps=[("prep", preprocessor), ("model", model)])


def cross_validate_model(pipe, X, y, name):
    """
    Corre validação cruzada estratificada (5 divisões).

    Devolve média e desvio-padrão de cada métrica. Um desvio-padrão baixo
    = modelo CONSISTENTE (dá resultados parecidos em qualquer subconjunto).
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["roc_auc", "recall", "precision", "f1", "accuracy"]
    results = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    summary = {}
    print(f"\n--- Validação cruzada (5 folds): {name} ---")
    for metric in scoring:
        scores = results[f"test_{metric}"]
        mean, std = scores.mean(), scores.std()
        summary[metric] = {"mean": round(float(mean), 4), "std": round(float(std), 4)}
        print(f"  {metric:10s}: {mean:.3f} ± {std:.3f}")
    return summary


def main():
    df = get_clean_data(save=False)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    # Divisão treino/teste (a mesma semente => comparável ao baseline)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    metrics_out = {}

    # ---- Baseline (Regressão Logística) para comparação ----
    logreg = Pipeline(steps=[
        ("prep", build_preprocessor(numeric_cols, categorical_cols)),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced",
                                     random_state=RANDOM_STATE)),
    ])
    metrics_out["baseline_logreg_cv"] = cross_validate_model(
        logreg, X, y, "Regressão Logística")

    # ---- HistGradientBoosting ----
    gb = build_gb_pipeline(preprocessor)
    metrics_out["gradient_boosting_cv"] = cross_validate_model(
        gb, X, y, "HistGradientBoosting")

    # Treinar no treino e avaliar no teste (holdout)
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    y_proba = gb.predict_proba(X_test)[:, 1]

    print("\n=== HistGradientBoosting — avaliação no conjunto de teste ===")
    print(classification_report(y_test, y_pred,
                                target_names=["Ficou (0)", "Saiu (1)"]))
    cm = confusion_matrix(y_test, y_pred)
    print("Matriz de confusão:\n", cm)

    metrics_out["gradient_boosting_holdout"] = {
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "confusion_matrix": cm.tolist(),
    }

    # Guardar modelo e métricas
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(gb, MODELS_DIR / "gradient_boosting_churn.joblib")

    metrics_path = ROOT / "reports" / "metrics_phase2.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2, ensure_ascii=False)

    print(f"\nModelo guardado em: {MODELS_DIR / 'gradient_boosting_churn.joblib'}")
    print(f"Métricas guardadas em: {metrics_path}")


if __name__ == "__main__":
    main()
