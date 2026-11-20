"""Metrics and plots shared by the baseline and the fine-tuned model."""
import matplotlib

matplotlib.use("Agg")  # save plots to files, no display window needed
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)

from src.config import FIGURES_DIR


def compute_metrics(eval_pred):
    """Accuracy + weighted F1. Called by the Hugging Face Trainer after each epoch."""
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }


def report(y_true, y_pred, labels, name):
    """Print a classification report and save a normalised confusion matrix."""
    print(f"\n===== {name} =====")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 9))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=labels, normalize="true",
        values_format=".2f", colorbar=False, ax=ax, xticks_rotation=90,
    )
    ax.set_title(f"Confusion matrix - {name}")
    fig.tight_layout()

    path = FIGURES_DIR / f"confusion_matrix_{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved confusion matrix -> {path}")
