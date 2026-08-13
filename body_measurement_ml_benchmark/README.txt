# Body Measurement ML Benchmark

This is a small, REAL-data benchmark assembled from the public sample of the
TrainingDataPro / UniqueData `body-measurements-dataset` on Hugging Face.

Targets:
- height_cm
- chest_cm
- waist_cm

Each row corresponds to a real subject record in the source sample. The source
dataset also provides front, side and selfie photographs for these subject IDs.

Important:
- `measurement_status=tbr` means the source labels that measurement with `_tbr`.
  Do NOT silently treat those as identical to directly measured values.
- No synthetic body measurements or AI-generated human photographs are included.
- The CSV keeps source-relative image paths so you can download the corresponding
  real images from the original dataset.
- This package is a benchmark subset, not a replacement for the complete source
  dataset.

Source:
https://huggingface.co/datasets/UniqueData/body-measurements-dataset

For a stronger evaluation, keep subjects completely separated between training
and test sets, and evaluate MAE/RMSE separately for height, chest and waist.
