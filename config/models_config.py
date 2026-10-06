"""Parâmetros centralizados dos modelos de classificação."""

# Random Forest -------------------------------------------------------------
RF_CONFIG = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": "balanced",   # compensa desbalanceamento sem precisar de SMOTE isolado
    "n_jobs": 1,                   # compatível com ambientes Windows restritos
    "random_state": 42,
}

# XGBoost -------------------------------------------------------------------
XGB_CONFIG = {
    "n_estimators": 400,           # Número de árvores que serão construídas
    "max_depth": 6,                # Profundidade máxima de cada árvore
    "learning_rate": 0.03,         # Taxa de aprendizado; menor = aprendizado mais gradual
    "subsample": 0.8,              # Usa 80% das amostras em cada árvore, ajudando a reduzir overfitting
    "colsample_bytree": 0.8,       # Usa 80% das features em cada árvore, ajudando a reduzir overfitting
    "min_child_weight": 3,         # Peso mínimo necessário para criar uma nova divisão em um nó
    "gamma": 0.1,                  # Ganho mínimo necessário para realizar uma divisão
    "reg_alpha": 0.1,              # Regularização L1; ajuda a reduzir complexidade e overfitting
    "reg_lambda": 0,             # Regularização L2; penaliza modelos muito complexos
    "eval_metric": "aucpr",       # Métrica de avaliação baseada na curva Precision-Recall; útil para classes desbalanceadas
    "tree_method": "hist",         # Método baseado em histogramas; geralmente mais rápido e eficiente
    "n_jobs": -1,                  # Usa todos os núcleos de CPU disponíveis
    "random_state": 42,            # Define a semente para obter resultados reproduzíveis

    # "scale_pos_weight" é definido dinamicamente após o reequilíbrio em train.py.
}

# LightGBM ------------------------------------------------------------------
LGBM_CONFIG = {
    "n_estimators": 400,
    "num_leaves": 31,
    "max_depth": -1,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "is_unbalance": True,
    "n_jobs": -1,
    "random_state": 42,
    "verbosity": -1,
}

# Proporções de oversampling comparadas dentro de cada fold.
SAMPLERS = ("smotenc_10", "smotenc_30", "smotenc_50",
            "random_over_10", "random_over_30", "random_over_50")

MODEL_REGISTRY_KEYS = ["random_forest", "xgboost", "lightgbm", "balanced_random_forest"]
