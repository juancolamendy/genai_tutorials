from typing import List, Dict, Any, Tuple


def confusion_matrix_binary(
    y_true: List[int], y_pred: List[int]
) -> Dict[str, int]:
    """
    Compute confusion matrix for binary classification.

    Assumes positive class is 1 and negative class is 0.
    Returns a dict with TP, FP, TN, FN counts.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    tp = fp = tn = fn = 0

    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            tp += 1
        elif true == 0 and pred == 1:
            fp += 1
        elif true == 0 and pred == 0:
            tn += 1
        elif true == 1 and pred == 0:
            fn += 1
        else:
            # If labels are not strictly 0/1, you can adjust or raise
            raise ValueError(f"Unsupported label values: true={true}, pred={pred}")

    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def classification_metrics_binary(
    y_true: List[int], y_pred: List[int]
) -> Dict[str, Any]:
    """
    Compute accuracy, precision, and recall for binary classification.

    Uses confusion_matrix_binary internally.
    """
    cm = confusion_matrix_binary(y_true, y_pred)
    tp, fp, tn, fn = cm["TP"], cm["FP"], cm["TN"], cm["FN"]

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0  # [web:78][web:79]

    # precision = TP / (TP + FP)
    precision_den = tp + fp
    precision = tp / precision_den if precision_den > 0 else 0.0  # [web:78][web:79]

    # recall = TP / (TP + FN)
    recall_den = tp + fn
    recall = tp / recall_den if recall_den > 0 else 0.0  # [web:78][web:79][web:82]

    return {
        "confusion_matrix": cm,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }


if __name__ == "__main__":
    # Example: 1 = positive class, 0 = negative class
    y_true = [1, 0, 1, 1, 0, 0, 1]
    y_pred = [1, 0, 0, 1, 0, 1, 1]

    metrics = classification_metrics_binary(y_true, y_pred)

    print("Confusion matrix:", metrics["confusion_matrix"])
    print("Accuracy:", metrics["accuracy"])
    print("Precision:", metrics["precision"])
    print("Recall:", metrics["recall"])
