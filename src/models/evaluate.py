from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, fbeta_score, precision_score, recall_score, roc_auc_score,
                             ConfusionMatrixDisplay)


def evaluate_predictions(y_true, probabilities, threshold):
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    return {
        "test_average_precision": float(average_precision_score(y_true, probabilities)),
        "test_roc_auc": float(roc_auc_score(y_true, probabilities)),
        "test_precision": float(precision_score(y_true, predictions, zero_division=0)),
        "test_recall": float(recall_score(y_true, predictions, zero_division=0)),
        "test_f1": float(f1_score(y_true, predictions, zero_division=0)),
        "test_f2": float(fbeta_score(y_true, predictions, beta=2, zero_division=0)),
        "test_accuracy": float(accuracy_score(y_true, predictions)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
        "predictions": predictions,
    }


def plot_confusion_matrix(y_true, predictions, metrics, output_path):
    """Save labeled matrix with requested summary metrics displayed beside it."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ConfusionMatrixDisplay.from_predictions(y_true, predictions, display_labels=["Sem AVC", "Com AVC"],
                                            cmap="Blues", values_format="d", colorbar=False, ax=ax)
    ax.set_title("Matriz de confusão — holdout")
    summary = (f"F2-score: {metrics['test_f2']:.3f}\n"
               f"F1-score: {metrics['test_f1']:.3f}\n"
               f"Accuracy: {metrics['test_accuracy']:.3f}\n"
               f"Precision: {metrics['test_precision']:.3f}\n"
               f"Recall: {metrics['test_recall']:.3f}")
    fig.text(0.73, 0.52, summary, va="center", fontsize=12,
             bbox={"boxstyle": "round,pad=0.7", "facecolor": "#f2f5f9", "edgecolor": "#aab7c4"})
    fig.tight_layout(rect=(0, 0, 0.72, 1))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
