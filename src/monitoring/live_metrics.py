"""Minimal privacy-preserving inference counters; does not persist patient features."""
import json
from datetime import datetime, timezone
from threading import Lock
from config.paths_config import LOGS_DIR

_lock = Lock()


def record_prediction(prediction, model_name):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    event = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
             "event": "prediction", "prediction": int(prediction), "model": model_name}
    with _lock, (LOGS_DIR / "inference_metrics.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event) + "\n")
