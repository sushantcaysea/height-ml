"""
Compare SMPLer-X digital measurements vs body_measurement_ml_benchmark ground truth.

1) Continuous metrics: abs error, % error, accuracy% = 100 - % error
2) Classification metrics (after binning cm into classes):
   confusion matrix, accuracy, precision, recall, F1
"""
import csv
from pathlib import Path

import numpy as np

try:
    from sklearn.metrics import (
        confusion_matrix,
        accuracy_score,
        precision_recall_fscore_support,
        f1_score,
    )
except ImportError:
    confusion_matrix = None

ROOT = Path(__file__).resolve().parent
BENCH = ROOT.parent / "body_measurement_ml_benchmark" / "body_measurement_benchmark.csv"

FRAME_TO_SUBJECT = {
    1: 0,
    2: 1,
    3: 10,
    4: 11,
    5: 12,
    6: 13,
    7: 14,
    8: 15,
    9: 16,
}

DIGITAL = {
    1: {"height_cm": 174.8, "chest_cm": 105.3, "waist_cm": 94.5},
    2: {"height_cm": 166.3, "chest_cm": 186.6, "waist_cm": 95.0},
    3: {"height_cm": 166.2, "chest_cm": 185.5, "waist_cm": 94.4},
    4: {"height_cm": 169.6, "chest_cm": 104.8, "waist_cm": 96.8},
    5: {"height_cm": 167.9, "chest_cm": 192.8, "waist_cm": 94.3},
    6: {"height_cm": 168.5, "chest_cm": 188.4, "waist_cm": 94.9},
    7: {"height_cm": 173.6, "chest_cm": 105.3, "waist_cm": 95.2},
    8: {"height_cm": 173.0, "chest_cm": 110.0, "waist_cm": 99.9},
    9: {"height_cm": 171.6, "chest_cm": 192.6, "waist_cm": 96.7},
}

# Bins: (name, low_inclusive, high_exclusive)
HEIGHT_BINS = [("short", 0, 165), ("medium", 165, 175), ("tall", 175, 300)]
CHEST_BINS = [("small", 0, 95), ("medium", 95, 110), ("large", 110, 300)]
WAIST_BINS = [("small", 0, 75), ("medium", 75, 95), ("large", 95, 300)]
BINS = {
    "height_cm": HEIGHT_BINS,
    "chest_cm": CHEST_BINS,
    "waist_cm": WAIST_BINS,
}


def load_gt():
    gt = {}
    with open(BENCH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = int(row["subject_id"])
            gt[sid] = {
                "height_cm": float(row["height_cm"]),
                "chest_cm": float(row["chest_cm"]),
                "waist_cm": float(row["waist_cm"]),
                "status": row["measurement_status"],
            }
    return gt


def metrics(pred, truth):
    err = abs(pred - truth)
    pct = err / truth * 100.0 if truth else float("nan")
    acc = 100.0 - pct
    return err, pct, acc


def bin_value(value, bins):
    for name, lo, hi in bins:
        if lo <= value < hi:
            return name
    return bins[-1][0]


def summarize(rows, title):
    print(f"\n=== {title} ===")
    if not rows:
        print("No rows.")
        return

    print(
        f"{'frame':>5} {'sid':>4} {'metric':<8} {'digital':>8} {'truth':>8} "
        f"{'abs_err':>8} {'pct_err':>8} {'acc%':>8} {'status':<9}"
    )
    print("-" * 80)

    by_metric = {"height_cm": [], "chest_cm": [], "waist_cm": []}
    for r in rows:
        print(
            f"{r['frame']:5d} {r['sid']:4d} {r['metric']:<8} {r['digital']:8.1f} {r['truth']:8.1f} "
            f"{r['abs_err']:8.1f} {r['pct_err']:8.1f} {r['acc']:8.1f} {r['status']:<9}"
        )
        by_metric[r["metric"]].append(r)

    print("-" * 80)
    print(f"{'SUMMARY':<20} {'MAE_cm':>8} {'mean_pct_err':>12} {'mean_acc%':>10}")
    for m, items in by_metric.items():
        mae = float(np.mean([x["abs_err"] for x in items]))
        mp = float(np.mean([x["pct_err"] for x in items]))
        ma = float(np.mean([x["acc"] for x in items]))
        print(f"{m:<20} {mae:8.1f} {mp:12.1f} {ma:10.1f}")

    all_acc = [x["acc"] for x in rows]
    print(f"\nOverall mean accuracy% (all metrics): {float(np.mean(all_acc)):.1f}%")


def classification_report_for_metric(rows, metric, bins, title):
    print(f"\n=== Classification: {title} / {metric} ===")
    labels = [b[0] for b in bins]
    y_true = []
    y_pred = []
    print(f"Bins: {bins}")
    print(f"{'sid':>4} {'truth_cm':>9} {'pred_cm':>9} {'true_cls':<8} {'pred_cls':<8}")
    for r in rows:
        if r["metric"] != metric:
            continue
        t_cls = bin_value(r["truth"], bins)
        p_cls = bin_value(r["digital"], bins)
        y_true.append(t_cls)
        y_pred.append(p_cls)
        print(f"{r['sid']:4d} {r['truth']:9.1f} {r['digital']:9.1f} {t_cls:<8} {p_cls:<8}")

    if not y_true:
        print("No rows for this metric.")
        return

    if confusion_matrix is None:
        print("Install scikit-learn: pip install scikit-learn")
        return

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    f1_macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)

    print("\nConfusion matrix (rows=truth/tape class, cols=predicted/digital class)")
    print("labels:", labels)
    print(cm)
    print(f"\nClassification accuracy: {acc:.3f}  ({acc*100:.1f}%)")
    print(f"F1 macro:    {f1_macro:.3f}")
    print(f"F1 weighted: {f1_weighted:.3f}")
    print(f"{'class':<8} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>8}")
    for i, lab in enumerate(labels):
        print(f"{lab:<8} {prec[i]:10.3f} {rec[i]:10.3f} {f1[i]:10.3f} {int(support[i]):8d}")


def main():
    gt = load_gt()
    all_rows = []
    measured_rows = []

    for frame, dig in DIGITAL.items():
        sid = FRAME_TO_SUBJECT[frame]
        truth = gt[sid]
        for metric in ("height_cm", "chest_cm", "waist_cm"):
            err, pct, acc = metrics(dig[metric], truth[metric])
            row = {
                "frame": frame,
                "sid": sid,
                "metric": metric,
                "digital": dig[metric],
                "truth": truth[metric],
                "abs_err": err,
                "pct_err": pct,
                "acc": acc,
                "status": truth["status"],
            }
            all_rows.append(row)
            if truth["status"] == "measured":
                measured_rows.append(row)

    summarize(all_rows, "ALL 9 subjects (includes tbr labels)")
    summarize(measured_rows, "ONLY status=measured (recommended continuous score)")

    print("\n" + "=" * 80)
    print("CLASSIFICATION METRICS (cm binned into classes)")
    print("Needed for confusion matrix / F1. Continuous cm accuracy is still the main score.")
    print("=" * 80)

    for metric, bins in BINS.items():
        classification_report_for_metric(
            measured_rows, metric, bins, "measured subset"
        )
        classification_report_for_metric(all_rows, metric, bins, "all subjects")

    print(
        """
HOW TO READ THIS
- Continuous accuracy% = 100 - |digital-truth|/truth*100   (best for body cm)
- Confusion matrix / F1 = after putting cm into bins (short/medium/tall etc.)
- Classification accuracy can look different from continuous accuracy%
- Chest F1 may be weak because digital chest is often overestimated (arms in slice)

Formulas:
  Precision = TP / (TP + FP)
  Recall    = TP / (TP + FN)
  F1        = 2 * precision * recall / (precision + recall)
"""
    )


if __name__ == "__main__":
    main()
