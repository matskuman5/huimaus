from sklearn.metrics import accuracy_score, f1_score, recall_score


def classification_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0, average="macro")
    sensitivity = recall_score(y_true, y_pred, zero_division=0, average="macro")
    return accuracy, f1, sensitivity


SEED = 33
