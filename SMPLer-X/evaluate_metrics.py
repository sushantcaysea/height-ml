"""
evaluate_metrics.py
-------------------
Compare digital measurements (from measure_body.py) to tape measurements.

Prints:
  1) Regression-style metrics (best for this project):
       - absolute error (cm)
       - percent error (%)
       - MAE
  2) Optional classification-style metrics (if course asks):
       - confusion matrix, accuracy, F1
     by putting height/chest/waist into size bins.

HOW TO USE
  1) Run measure_body.py first (or fill DIGITAL below with those numbers).
  2) Edit TAPE with real tape-measure values for the same person.
  3) conda activate smplerx
     python evaluate_metrics.py
"""

import numpy as np

try:
    from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
except ImportError:
    confusion_matrix = accuracy_score = f1_score = None


# =============================================================================
# EDIT THESE NUMBERS
# =============================================================================

# From your measure_body.py AVERAGE (or per-photo list)
DIGITAL = {
    "height_cm": 174.1,
    "chest_cm": 110.0,
    "waist_cm": 96.1,
}

# Fill with real tape measurements of the SAME person (cm).
# If you don't have tape yet, leave as None and only digital is shown.
TAPE = {
    "height_cm": None,  # e.g. 172.0
    "chest_cm": None,   # e.g. 98.0
    "waist_cm": None,   # e.g. 84.0
}

# Optional: multiple photos' digital values for MAE across samples
# Leave empty [] to use only DIGITAL averages above.
DIGITAL_PER_PHOTO = [
    # {"height_cm": 175.3, "chest_cm": 113.2, "waist_cm": 97.6},
    # {"height_cm": 175.1, "chest_cm": 108.8, "waist_cm": 95.9},
]

# Bins for optional confusion-matrix demo (edit if you want)
# label: (low_inclusive, high_exclusive)  last bin high can be big
HEIGHT_BINS = [("short", 0, 165), ("medium", 165, 175), ("tall", 175, 300)]
CHEST_BINS = [("small", 0, 95), ("medium", 95, 105), ("large", 105, 200)]
WAIST_BINS = [("small", 0, 80), ("medium", 80, 95), ("large", 95, 200)]


# =============================================================================
# Helpers
# =============================================================================

def bin_value(value, bins):
    """Put a continuous cm value into a class name using bins."""
    for name, lo, hi in bins:
        if lo <= value < hi:
            return name
    return bins[-1][0]


def print_error_table(digital, tape):
    print("\n=== Measurement error (digital vs tape) ===")
    print(f"{'metric':<12} {'digital':>10} {'tape':>10} {'abs_err':>10} {'pct_err%':>10}")
    print("-" * 56)

    abs_errors = []
    for key in ("height_cm", "chest_cm", "waist_cm"):
        d = digital[key]
        t = tape.get(key)
        if t is None:
            print(f"{key:<12} {d:10.1f} {'—':>10} {'—':>10} {'—':>10}")
            continue
        abs_err = abs(d - t)
        pct_err = abs_err / t * 100.0 if t != 0 else float("nan")
        abs_errors.append(abs_err)
        print(f"{key:<12} {d:10.1f} {t:10.1f} {abs_err:10.1f} {pct_err:10.1f}")

    if abs_errors:
        mae = float(np.mean(abs_errors))
        print("-" * 56)
        print(f"{'MAE (cm)':<12} {mae:10.1f}")
        print("(MAE = mean absolute error over height/chest/waist)")
    else:
        print("\nNo tape values set yet. Edit TAPE = {...} at the top of this file.")


def print_classification_metrics(digital, tape):
    print("\n=== Optional: confusion matrix / accuracy / F1 ===")
    print("(Only meaningful after you set TAPE. Values are binned into classes.)")

    if any(tape.get(k) is None for k in ("height_cm", "chest_cm", "waist_cm")):
        print("Skip: fill all TAPE height/chest/waist first.")
        return

    if confusion_matrix is None:
        print("Skip: install scikit-learn →  pip install scikit-learn")
        return

    # One "sample" with 3 attributes treated as 3 classification decisions.
    # For a real study you'd have many people; here we demo the metrics API.
    specs = [
        ("height", digital["height_cm"], tape["height_cm"], HEIGHT_BINS),
        ("chest", digital["chest_cm"], tape["chest_cm"], CHEST_BINS),
        ("waist", digital["waist_cm"], tape["waist_cm"], WAIST_BINS),
    ]

    y_true = []
    y_pred = []
    labels_order = []

    print("\nBinned labels:")
    for name, d, t, bins in specs:
        true_lab = bin_value(t, bins)
        pred_lab = bin_value(d, bins)
        print(f"  {name}: tape={t:.1f}→{true_lab}, digital={d:.1f}→{pred_lab}")
        y_true.append(true_lab)
        y_pred.append(pred_lab)
        for lab, _, _ in bins:
            if lab not in labels_order:
                labels_order.append(lab)

    # With only 3 decisions this is a TOY demo — still shows the formulas.
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, labels=labels_order, average="macro", zero_division=0)

    print("\nLabel order:", labels_order)
    print("Confusion matrix (rows=tape/true, cols=digital/pred):")
    print(cm)
    print(f"Accuracy: {acc:.3f}   (= correct / total)")
    print(f"F1 macro: {f1:.3f}   (= average F1 across classes)")
    print("""
Formulas (classification):
  Accuracy = (TP-style correct count) / N
  Precision_c = TP_c / (TP_c + FP_c)
  Recall_c    = TP_c / (TP_c + FN_c)
  F1_c        = 2 * precision_c * recall_c / (precision_c + recall_c)
  F1 macro    = average of F1_c over classes
""")


def print_mae_across_photos(per_photo, tape):
    if not per_photo or any(tape.get(k) is None for k in ("height_cm", "chest_cm", "waist_cm")):
        return
    print("\n=== MAE across photos (vs same tape) ===")
    for key in ("height_cm", "chest_cm", "waist_cm"):
        errs = [abs(p[key] - tape[key]) for p in per_photo]
        print(f"  {key}: MAE={np.mean(errs):.1f} cm  (n={len(errs)})")


def main():
    print("Digital measurements used:")
    for k, v in DIGITAL.items():
        print(f"  {k}: {v}")

    print_error_table(DIGITAL, TAPE)
    print_mae_across_photos(DIGITAL_PER_PHOTO, TAPE)
    print_classification_metrics(DIGITAL, TAPE)

    print("\nNOTE: For body measurements, absolute/percent error and MAE matter most.")
    print("Confusion matrix / accuracy / F1 need class bins and are secondary here.")


if __name__ == "__main__":
    main()
