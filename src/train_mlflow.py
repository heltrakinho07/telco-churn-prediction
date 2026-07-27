"""
train_mlflow.py — Treino com tracking de experiências (MLflow).

Fase 4 (MLOps). O MLflow regista cada treino: que parâmetros usámos, que
métricas obtivemos e o modelo resultante. Assim consegues comparar
experiências ao longo do tempo, em vez de andar a apontar números à mão.

Correr:
    python src/train_mlflow.py
Ver os resultados numa interface web:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
    (depois abrir http://localhost:5000)
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import mlflow
import mlflow.sklearn
from config import RANDOM_STATE, ROOT, TARGET
from data_prep import get_clean_data

# Hiperparâmetros da experiência (num só sítio, fáceis de mexer)
PARAMS = {
    "max_iter": 300,
    "learning_rate": 0.05,
    "max_depth": 4,
    "l2_regularization": 1.0,
    "class_weight": "balanced",
}


def main():
    # Guardar as experiências numa base de dados SQLite local do projeto.
    # (O MLflow recente já não recomenda o armazenamento em ficheiros soltos.)
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("churn-telco")

    df = get_clean_data(save=False)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])
    model = HistGradientBoostingClassifier(random_state=RANDOM_STATE, **PARAMS)
    pipe = Pipeline([("prep", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    with mlflow.start_run(run_name="histgradientboosting"):
        # 1) Registar os parâmetros usados
        mlflow.log_params(PARAMS)
        mlflow.log_param("modelo", "HistGradientBoosting")

        # 2) Consistência: ROC-AUC médio em validação cruzada (5 folds)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
        mlflow.log_metric("cv_roc_auc_media", float(cv_auc.mean()))
        mlflow.log_metric("cv_roc_auc_desvio", float(cv_auc.std()))

        # 3) Treinar e avaliar no holdout
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        mlflow.log_metric("holdout_roc_auc", float(roc_auc_score(y_test, y_proba)))
        mlflow.log_metric("holdout_recall", float(recall_score(y_test, y_pred)))
        mlflow.log_metric("holdout_f1", float(f1_score(y_test, y_pred)))

        # 4) Registar o próprio modelo como artefacto
        mlflow.sklearn.log_model(pipe, name="modelo")

        print("Experiência registada no MLflow.")
        print(f"  CV ROC-AUC: {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")
        print(f"  Holdout ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")
        print(f"\nVer resultados:  cd {ROOT}  &&  mlflow ui")


if __name__ == "__main__":
    main()
