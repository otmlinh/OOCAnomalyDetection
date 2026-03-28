import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

def compute_metrics(y_true, y_score):
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score).astype(np.float32)

    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)

    fpr, tpr, thr = roc_curve(y_true, y_score)

    # FPR@95TPR
    idx = np.where(tpr >= 0.95)[0]
    fpr95 = float(fpr[idx[0]]) if len(idx) > 0 else float("nan")

    # TPR@5%FPR
    idx2 = np.where(fpr <= 0.05)[0]
    tpr_at_5fpr = float(tpr[idx2[-1]]) if len(idx2) > 0 else float("nan")

    # EER (xấp xỉ)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)

    return {
        "AUROC": float(auroc),
        "AUPRC": float(auprc),
        "FPR@95TPR": fpr95,
        "TPR@5%FPR": tpr_at_5fpr,
        "EER": eer,
        "N": int(len(y_true)),
    }
