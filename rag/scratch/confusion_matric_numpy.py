import numpy as np
from typing import Tuple


def confusion_matrix_binary_and(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[int, int, int, int]:
    """
    Compute binary confusion matrix (TP, FP, TN, FN) using NumPy
    logical masks and the & operator only.

    Assumes labels are 0/1 (or booleans), positive class = 1.
    """

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    # Ensure boolean arrays
    true_pos_class = (y_true == 1)
    pred_pos_class = (y_pred == 1)

    # Logical masks with & [web:84]
    tp = np.sum(true_pos_class & pred_pos_class)                      # true = 1, pred = 1
    fp = np.sum((y_true == 0) & pred_pos_class)                       # true = 0, pred = 1
    tn = np.sum((y_true == 0) & (y_pred == 0))                        # true = 0, pred = 0
    fn = np.sum(true_pos_class & (y_pred == 0))                       # true = 1, pred = 0

    return tp, fp, tn, fn


if __name__ == "__main__":
    y_true = np.array([1, 0, 1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 0, 1, 1])

    tp, fp, tn, fn = confusion_matrix_binary_and(y_true, y_pred)
    print("TP:", tp, "FP:", fp, "TN:", tn, "FN:", fn)

    # Optional: arrange into a 2x2 matrix (rows=true, cols=pred)
    cm = np.array([[tn, fp],
                   [fn, tp]])
    print("Confusion matrix:\n", cm)
