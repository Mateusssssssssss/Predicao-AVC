import json
import joblib
from config.paths_config import MODELS_DIR
from config.training_config import MODEL_SELECTION_METRIC, THRESHOLD_OBJECTIVE


def save_model_bundle(pipeline, threshold, features, model_name, sampler, metrics):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {"pipeline": pipeline, "threshold": threshold, "features": list(features),
              "model_name": model_name, "sampler": sampler, "metrics": metrics}
    joblib.dump(bundle, MODELS_DIR / "stroke_pipeline.joblib")
    card = {"selected_model": model_name, "sampler": sampler, "threshold": threshold,
            "selection_metric": MODEL_SELECTION_METRIC,
            "threshold_objective": THRESHOLD_OBJECTIVE, "metrics": metrics}
    (MODELS_DIR / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
