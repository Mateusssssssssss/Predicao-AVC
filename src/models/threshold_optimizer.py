import numpy as np
from sklearn.metrics import precision_recall_curve
from config.training_config import THRESHOLD_BETA, THRESHOLD_DEFAULT, THRESHOLD_OBJECTIVE


def find_best_threshold(y_true, probabilities, objective=THRESHOLD_OBJECTIVE):
    """Search the threshold that maximizes the configured F1 or F2 objective."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if not len(thresholds):
        return THRESHOLD_DEFAULT
    beta = THRESHOLD_BETA if objective == "f2" else 1.0
    if objective not in {"f1", "f2"}:
        raise ValueError("objective deve ser 'f1' ou 'f2'")
    beta_squared = beta ** 2
    best_threshold, best_score = THRESHOLD_DEFAULT, -1.0
    for threshold, p, r in zip(thresholds, precision[:-1], recall[:-1]):
        score = (1 + beta_squared) * p * r / (beta_squared * p + r + 1e-12)
        if np.isfinite(score) and score > best_score:
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold
