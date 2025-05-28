import numpy as np


def compute_robust_threshold(
    scores: np.array, k: int, epsilon: float, alpha: float, min_value: float = 0
) -> float:
    """

    Compute the certifiably robust calibration threshold again calibration feature poisoning.

    args:
    - scores: np.array: Containing the calibration scores for the true labels.
    - k: int: number of poisoned samples.
    - epsilon: float: amount of poisoning in l_2 norm.
    - alpha: float: conformal coverage level.
    - min_value: float: minimum calibration score value (0 for sigmoid or softmax for example).

    returns:
    - lbd: float: the certifiably robust calibration threshold.
    """
    n_cal = len(scores) + 1
    idxs_sort = np.argsort(scores)
    idx_threshold = np.floor(alpha * n_cal).astype(int)

    seen = []
    remaining_k = k

    while remaining_k >= 0:
        if idxs_sort[idx_threshold] in seen:
            scores[idxs_sort[idx_threshold + 1]] -= epsilon
            return np.sort(scores)[idx_threshold]
        seen.append(idxs_sort[idx_threshold])

        scores[idxs_sort[idx_threshold]] -= epsilon
        idxs_sort = idxs_sort[np.argsort(scores[idxs_sort])]
        remaining_k -= 1

    return np.sort(scores)[idx_threshold]
