"""Evaluation metrics shared by every model in the project.

The primary metric is top-k capture: the share of all converting customers in
the test set that land in the top `frac` of the ranked list. It is the number a
sales team calling the best-ranked quotes first actually experiences.
"""
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score


def top_k_capture(y_true, y_score, frac=0.10):
    """Fraction of all positives that fall in the top `frac` of rows by score.

    Ties at the cutoff are split proportionally (the expected value under random
    tie-breaking), so models with coarse scores, such as small XGBoost
    ensembles, are not helped or hurt by row order.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    k = int(np.ceil(frac * len(y_true)))
    cutoff = np.sort(y_score)[::-1][k - 1]
    above = y_score > cutoff
    at = y_score == cutoff
    taken_at_cutoff = k - above.sum()
    captured = y_true[above].sum() + y_true[at].sum() * taken_at_cutoff / at.sum()
    return captured / y_true.sum()


def capture_ceiling(y_true, frac=0.10):
    """Best achievable top-k capture: about 0.53 at frac=0.10 with 19% positives."""
    y_true = np.asarray(y_true, dtype=float)
    k = int(np.ceil(frac * len(y_true)))
    return min(k, y_true.sum()) / y_true.sum()


def evaluate(y_true, y_prob, frac=0.10):
    """All reported metrics for one set of predicted conversion probabilities."""
    capture = top_k_capture(y_true, y_prob, frac)
    return {
        "capture": capture,
        "capture_of_ceiling": capture / capture_ceiling(y_true, frac),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob, labels=[0, 1]),
    }
