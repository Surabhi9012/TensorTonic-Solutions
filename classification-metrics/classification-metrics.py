import numpy as np

def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    """
    Compute accuracy, precision, recall, F1 for single-label classification.
    Averages:
        - micro
        - macro
        - weighted
        - binary
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    accuracy = np.mean(y_true == y_pred)

    labels = np.unique(np.concatenate([y_true, y_pred]))

    precisions = []
    recalls = []
    f1s = []
    supports = []

    for label in labels:

        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        support = np.sum(y_true == label)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1s = np.array(f1s)
    supports = np.array(supports)

    # Binary averaging
    if average == "binary":

        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
        fn = np.sum((y_true == pos_label) & (y_pred != pos_label))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

    # Micro averaging
    elif average == "micro":

        tp = np.sum(y_true == y_pred)
        fp = np.sum(y_true != y_pred)
        fn = fp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

    # Macro averaging
    elif average == "macro":

        precision = np.mean(precisions)
        recall = np.mean(recalls)
        f1 = np.mean(f1s)

    # Weighted averaging
    elif average == "weighted":

        total_support = np.sum(supports)

        precision = np.sum(precisions * supports) / total_support
        recall = np.sum(recalls * supports) / total_support
        f1 = np.sum(f1s * supports) / total_support

    else:
        raise ValueError(
            "average must be one of: micro, macro, weighted, binary"
        )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }