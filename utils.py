from sklearn.metrics import accuracy_score, f1_score, recall_score
import numpy as np


def classification_metrics(y_true, y_pred, class_labels=None):
    """
    Compute classification metrics matching the manuscript's Section 5.3.

    Returns:
        accuracy: Overall accuracy (Eq. 5)
        f1_macro: Unweighted macro F1 score (Eq. 6)
        per_class_sens: Per-class sensitivities as dict {label: sensitivity} (Eq. 4)
    """
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, zero_division=0, average="macro")

    # Get per-class sensitivities
    if class_labels is None:
        class_labels = np.unique(y_true)

    per_class_sens = recall_score(
        y_true, y_pred, labels=class_labels, zero_division=0, average=None
    )

    # Return as dict mapping class label to sensitivity
    sens_dict = {label: sens for label, sens in zip(class_labels, per_class_sens)}

    return accuracy, f1_macro, sens_dict


SEED = 33
