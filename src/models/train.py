"""Cross-validation, model comparison, threshold selection and artifact packaging."""
import json
import logging
import warnings
from logging.handlers import RotatingFileHandler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import (StratifiedKFold, cross_val_predict, cross_validate,
                                    train_test_split)
from sklearn.metrics import precision_recall_curve, roc_curve
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier

from config.models_config import LGBM_CONFIG, RF_CONFIG, SAMPLERS, XGB_CONFIG
from config.paths_config import DATA_PATH, FIGURES_DIR, LOGS_DIR, MLFLOW_DB, REPORTS_DIR
from config.training_config import (CV_FOLDS, MLFLOW_EXPERIMENT, RANDOM_STATE,
                                    TARGET_COLUMN, TEST_SIZE, THRESHOLD_OBJECTIVE,
                                    MODEL_SELECTION_METRIC)
from src.data.balancing import build_sampler
from src.data.loader import load_dataset
from src.data.preprocessing import (CATEGORICAL_FEATURES, NUMERIC_FEATURES,
                                    build_preprocessor)
from src.features.engineering import split_features_target
from src.models.evaluate import evaluate_predictions, plot_confusion_matrix
from src.models.registry import save_model_bundle
from src.models.threshold_optimizer import find_best_threshold

LOGGER = logging.getLogger("stroke_training")
if not LOGGER.handlers:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(LOGS_DIR / "training.log", maxBytes=2_000_000,
                                  backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


def build_models(scale_pos_weight=None):
    models = {"random_forest": RandomForestClassifier(**RF_CONFIG),
              "xgboost": XGBClassifier(**XGB_CONFIG)}
    if scale_pos_weight is not None:
        # Train this candidate on the natural distribution, without resampling.
        models["xgboost_weighted"] = XGBClassifier(
            **XGB_CONFIG, scale_pos_weight=float(scale_pos_weight))
    try:
        from lightgbm import LGBMClassifier
        models["lightgbm"] = LGBMClassifier(**LGBM_CONFIG)
    except ImportError:
        warnings.warn("LightGBM não instalado; modelo ignorado.")
    models["balanced_random_forest"] = BalancedRandomForestClassifier(
        n_estimators=RF_CONFIG["n_estimators"], max_depth=RF_CONFIG["max_depth"],
        min_samples_split=RF_CONFIG["min_samples_split"],
        min_samples_leaf=RF_CONFIG["min_samples_leaf"],
        max_features=RF_CONFIG["max_features"], sampling_strategy="all",
        replacement=True, bootstrap=False, n_jobs=1, random_state=RF_CONFIG["random_state"])
    return models


def build_pipeline(model, sampler_name, feature_columns=None):
    encoding = "onehot" if sampler_name == "none_onehot" else "ordinal"
    numeric = NUMERIC_FEATURES if feature_columns is None else [
        name for name in NUMERIC_FEATURES if name in feature_columns]
    categorical = CATEGORICAL_FEATURES if feature_columns is None else [
        name for name in CATEGORICAL_FEATURES if name in feature_columns]
    return Pipeline([("preprocess", build_preprocessor(
                         categorical_encoding=encoding, numeric_features=numeric,
                         categorical_features=categorical)),
                     ("sampler", build_sampler(sampler_name, numeric, categorical)),
                     ("model", model)])


def _log_to_mlflow(model_name, sampler_name, result):
    try:
        import mlflow
    except ImportError:
        warnings.warn("MLflow não instalado; tracking ignorado.")
        return
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    metrics = {key: value for key, value in result.items()
               if key.startswith(("cv_", "test_")) and isinstance(value, (int, float))}
    with mlflow.start_run(run_name=f"{model_name}-{sampler_name}"):
        mlflow.log_params({"model": model_name, "sampler": sampler_name,
                           "categorical_encoding": result.get("categorical_encoding", "ordinal"),
                           "threshold": result["threshold"], "cv_folds": CV_FOLDS,
                           "random_state": RANDOM_STATE,
                           "threshold_objective": THRESHOLD_OBJECTIVE})
        mlflow.log_metrics(metrics)


def run_training(data_path=DATA_PATH, run_mlflow=True):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_dataset(data_path)
    X, y = split_features_target(frame)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    selected_features = list(X_train.columns)
    LOGGER.info("features usadas (id excluído): %s", selected_features)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"average_precision": "average_precision", "roc_auc": "roc_auc",
               "recall": "recall", "f1": "f1",
               "f2": make_scorer(fbeta_score, beta=2, zero_division=0)}
    results, fitted, test_probabilities = [], {}, {}
    # Peso de classe é comparado no conjunto natural, sem oversampling.
    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    for model_name, model in build_models(scale_pos_weight).items():
        # Compare all sampler ratios with the same XGBoost estimator; weights stay separate.
        sampler_names = ("none", *SAMPLERS) if model_name == "xgboost" else ("none",)
        for sampler_name in sampler_names:
            pipeline = build_pipeline(model, sampler_name, selected_features)
            cv_results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring,
                                        n_jobs=1, return_train_score=False)
            oof_probabilities = cross_val_predict(pipeline, X_train, y_train, cv=cv,
                                                  method="predict_proba", n_jobs=1)[:, 1]
            threshold = find_best_threshold(y_train, oof_probabilities)
            trained = build_pipeline(model, sampler_name, selected_features).fit(X_train, y_train)
            probabilities = trained.predict_proba(X_test)[:, 1]
            metrics = evaluate_predictions(y_test, probabilities, threshold)
            result = {
                "model": model_name, "sampler": sampler_name, "threshold": threshold,
                "categorical_encoding": "onehot" if sampler_name == "none_onehot" else "ordinal",
                "cv_average_precision_mean": float(cv_results["test_average_precision"].mean()),
                "cv_average_precision_std": float(cv_results["test_average_precision"].std()),
                "cv_roc_auc_mean": float(cv_results["test_roc_auc"].mean()),
                "cv_recall_mean": float(cv_results["test_recall"].mean()),
                "cv_f2_mean": float(cv_results["test_f2"].mean()),
                **{key: value for key, value in metrics.items() if key != "predictions"},
            }
            LOGGER.info("modelo=%s sampler=%s cv_auprc=%.4f holdout_auprc=%.4f threshold=%.4f",
                        model_name, sampler_name, result["cv_average_precision_mean"],
                        result["test_average_precision"], threshold)
            results.append(result)
            fitted[(model_name, sampler_name)] = trained
            test_probabilities[(model_name, sampler_name)] = probabilities
            if run_mlflow:
                _log_to_mlflow(model_name, sampler_name, result)

    # cv_f2_mean é o scorer de CV no threshold padrão; holdout fica fora da seleção.
    best = max(results, key=lambda item: item[MODEL_SELECTION_METRIC])
    best_key = (best["model"], best["sampler"])
    LOGGER.info("selecionado modelo=%s sampler=%s threshold=%.4f",
                best["model"], best["sampler"], best["threshold"])
    best_pipeline = fitted[best_key]
    best_probabilities = test_probabilities[best_key]
    best_predictions = (best_probabilities >= best["threshold"]).astype(int)
    save_model_bundle(best_pipeline, best["threshold"], selected_features,
                      best["model"], best["sampler"],
                      {key: value for key, value in best.items() if key.startswith(("cv_", "test_"))})

    (REPORTS_DIR / "model_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    plot_confusion_matrix(y_test, best_predictions, best,
                          FIGURES_DIR / "confusion_matrix.png")
    fpr, tpr, _ = roc_curve(y_test, best_probabilities)
    precision, recall, _ = precision_recall_curve(y_test, best_probabilities)
    fig, ax = plt.subplots(); ax.plot(fpr, tpr, label=f"AUC={best['test_roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey"); ax.set(xlabel="FPR", ylabel="TPR", title="Curva ROC — holdout")
    ax.legend(); fig.tight_layout(); fig.savefig(FIGURES_DIR / "roc_curve.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(); ax.plot(recall, precision, label=f"AUPRC={best['test_average_precision']:.3f}")
    ax.axhline(float(y_test.mean()), linestyle="--", color="grey", label="baseline prevalência")
    ax.set(xlabel="Recall", ylabel="Precisão", title="Curva Precision-Recall — holdout")
    ax.legend(); fig.tight_layout(); fig.savefig(FIGURES_DIR / "precision_recall_curve.png", dpi=160); plt.close(fig)
    return best, results


if __name__ == "__main__":
    best, _ = run_training()
    print(json.dumps(best, indent=2))
