# Option A — Body Measurement Mini-Pipeline
## Project Write-up

**Goal:** Use a pretrained 3D human pose/shape model (SMPLer-X) on photos, extract at least three body measurements (height, chest, waist), and document setup, troubleshooting, and validation against ground-truth labels (no lab 3D scanner).

**Time box:** ~4–6 hours (plus Windows debugging and benchmark evaluation)

---

## 1. What we built (pipeline overview)

```
Photos (.jpg)
  → mmdet (person box / crop focus)
  → SMPLer-X (predict shape betas + pose)
  → SMPL-X (build 3D mesh points)
  → save .npz / .obj
  → measure_body.py (NumPy)
  → height, chest, waist (cm)
  → compute_accuracy.py (compare to labeled benchmark)
```

### Who does what

| Component | Role |
|-----------|------|
| Input images | Photos of a standing person (we used front views) |
| mmdet | Finds where the person is (bounding box) |
| SMPLer-X | AI that guesses body shape (`betas`) and pose |
| SMPL-X | Body template that turns betas/pose into 3D mesh points |
| SMPL | Older body model; required by the repo alongside SMPL-X |
| `measure_body.py` | Rebuild T-pose from betas; measure height / chest / waist |
| `compute_accuracy.py` | Compare digital cm vs benchmark ground truth |

**Important distinction**

- **SMPLer-X** = the AI (brain)
- **SMPL-X** = the 3D body doll (template)
- They are not alternatives; SMPLer-X *uses* SMPL-X.

Photos do **not** need a real-world T-pose. People stand normally. We only build a **digital T-pose** later so photo pose does not dominate the tape-around-mesh step.

---

## 2. Environment setup

### Hardware / software

- Windows 10/11
- NVIDIA GeForce RTX 3050
- Anaconda (Python 3.8 env `smplerx`)
- Git

### Steps we followed

1. **Clone the repo**
   ```text
   git clone https://github.com/MotrixLab/SMPLer-X.git
   ```
   Location: `C:\Users\Asus\Desktop\demo ml\SMPLer-X`

2. **Create conda environment**
   ```text
   conda create -n smplerx python=3.8 -y
   conda activate smplerx
   ```

3. **Install dependencies** (as in SMPLer-X README)
   - PyTorch 1.12.0 + CUDA 11.3 toolkit
   - `mmcv-full==1.7.1`
   - `pip install -r requirements.txt`
   - Editable install of `main/transformer_utils` (mmpose)
   - `pip install numpy==1.23.5` (compatibility fix)

4. **Download body models**
   - **SMPL-X v1.1 (NPZ+PKL)** from https://smpl-x.is.tue.mpg.de/
   - **SMPL Python v1.1.0** (male/female/neutral) from https://smpl.is.tue.mpg.de/
   - Place under:
     ```text
     common/utils/human_model_files/smplx/
     common/utils/human_model_files/smpl/
     ```
   - Extra SMPL-X helpers in `smplx/`:
     - `SMPLX_to_J14.pkl`
     - `MANO_SMPLX_vertex_ids.pkl`
     - `SMPL-X__FLAME_vertex_ids.npy`

5. **Download pretrained weights**
   - `smpler_x_s32.pth.tar` → `pretrained_models/`
   - mmdet Faster R-CNN `.pth` + config `.py` → `pretrained_models/mmdet/`

6. **Prepare photos for inference**
   - Folder: `demo/images/my_person/`
   - Names must be `000001.jpg`, `000002.jpg`, … (6-digit)
   - Final evaluation used **front standing images** from the local benchmark (see Section 5)

7. **Run inference**
   ```text
   cd main
   python inference.py --num_gpus 1 --exp_name output/demo_benchmark ^
     --pretrained_model smpler_x_s32 --agora_benchmark agora_model ^
     --img_path ../demo/images/my_person --start 1 --end 9 ^
     --output_folder ../demo/results/my_person ^
     --show_verts --show_bbox --save_mesh
   ```

8. **Extract measurements + score accuracy**
   ```text
   cd ..
   python measure_body.py
   python compute_accuracy.py
   ```

---

## 3. Key files (what each one is for)

### Tools / models we downloaded

| Path | Purpose | Has true height/chest/waist? |
|------|---------|------------------------------|
| SMPLer-X repo | Code to run the pipeline | No |
| `pretrained_models/smpler_x_s32.pth.tar` | SMPLer-X AI weights | No |
| `pretrained_models/mmdet/*` | Person detector | No |
| `human_model_files/smplx/` | SMPL-X body template | No |
| `human_model_files/smpl/` | SMPL body template | No |

### Created by inference

| Path | Purpose |
|------|---------|
| `demo/results/my_person/smplx/*.npz` | Predicted `betas`, pose, etc. |
| `demo/results/my_person/mesh/*.obj` | Posed 3D mesh |
| `demo/results/my_person/meta/*.json` | BBox / camera metadata |

### Our scripts

| Path | Purpose |
|------|---------|
| `measure_body.py` | `betas` → T-pose mesh → height/chest/waist |
| `evaluate_metrics.py` | Generic digital-vs-tape helper |
| `compute_accuracy.py` | Score vs `body_measurement_ml_benchmark` |

### Ground-truth data used for validation

| Path | Purpose |
|------|---------|
| `../body_measurement_ml_benchmark/` | Labeled height/chest/waist + real photos |
| `body_measurement_benchmark.csv` | Ground-truth cm values |
| `images/*/front_img.jpg` | Standing front photos used as input |

We also inspected SHAPY **Model Agency Data** (JSON labels). Many image URLs are dead (`not a jpeg`), so it was not usable for end-to-end photo validation. The local Hugging Face–based benchmark was usable instead.

---

## 4. How measurements work (simple math)

### Betas
- 10 numbers describing **body shape** (not pose).
- Predicted by SMPLer-X from the photo; stored in `.npz`.
- We rebuild a **digital T-pose** mesh from betas (pose = 0) before measuring.

### Height
- Mesh points `(x, y, z)` in meters; **Y is up**.
- `height = max(y) - min(y)` (head − feet), then ×100 for cm.

### Chest / waist
1. Choose slice height as a fraction of body height from the feet:
   - chest ≈ **74%** of height
   - waist ≈ **58%** of height  
   (approximate anatomy heuristics — not predicted by the AI)
2. Keep mesh points near that height (thin horizontal slice).
3. Project to `(x, z)` and measure outline length (circumference).
4. Outline length = sum of edge lengths around the outer ring (convex hull).

**Note:** Fixed percentages are a mini-pipeline shortcut. A stronger system would use mesh landmarks (e.g. underbust / navel).

---

## 5. Validation dataset and digital results

### Dataset
`body_measurement_ml_benchmark` (from public sample of UniqueData body-measurements dataset on Hugging Face):

- Source viewer: https://huggingface.co/datasets/UniqueData/body-measurements-dataset/viewer/default/train
- Real subject records with `height_cm`, `chest_cm`, `waist_cm`
- Front / side / selfie images (we used **front** only)
- **We keep / use `measurement_status` as the label-quality flag** (`measured` vs `tbr`)

Important distinction on the Hugging Face viewer page:
- The viewer column **`label`** is only the **subject / folder id** (0, 1, 10, 11, …), not measurement quality.
- Example: [train row 72](https://huggingface.co/datasets/UniqueData/body-measurements-dataset/viewer/default/train?row=72) shows `label = 0` → subject folder `images/0/`, not a status string.
- True quality comes from each subject’s `measurements.json`: values ending in `_tbr` are weaker. Our CSV sets:
  - `measurement_status = measured` when height/chest/waist are clean numbers
  - `measurement_status = tbr` when key circumferences are marked `_tbr`
- For reported accuracy we **prefer `measurement_status = measured`** and still show full-set numbers separately.

### Image mapping used

| Inference file | Subject ID | Truth H | Truth chest | Truth waist | Label status |
|----------------|------------|---------|-------------|-------------|--------------|
| 000001.jpg | 0 | 159 | 79 | 68 | measured |
| 000002.jpg | 1 | 157 | 93 | 79 | measured |
| 000003.jpg | 10 | 163 | 112 | 88 | tbr |
| 000004.jpg | 11 | 163 | 86 | 65 | measured |
| 000005.jpg | 12 | 165 | 90 | 73 | measured |
| 000006.jpg | 13 | 155 | 105 | 102 | tbr |
| 000007.jpg | 14 | 172 | 86 | 77 | tbr |
| 000008.jpg | 15 | 168 | 103 | 101 | tbr |
| 000009.jpg | 16 | 175 | 107 | 92 | measured |

### Digital outputs (`measure_body.py`)

| File | Height (cm) | Chest (cm) | Waist (cm) |
|------|-------------|------------|------------|
| 00001_0.npz | 174.8 | 105.3 | 94.5 |
| 00002_0.npz | 166.3 | 186.6 | 95.0 |
| 00003_0.npz | 166.2 | 185.5 | 94.4 |
| 00004_0.npz | 169.6 | 104.8 | 96.8 |
| 00005_0.npz | 167.9 | 192.8 | 94.3 |
| 00006_0.npz | 168.5 | 188.4 | 94.9 |
| 00007_0.npz | 173.6 | 105.3 | 95.2 |
| 00008_0.npz | 173.0 | 110.0 | 99.9 |
| 00009_0.npz | 171.6 | 192.6 | 96.7 |

---

## 6. Evaluation metrics (digital vs labeled ground truth)

We report two layers of metrics from `compute_accuracy.py`:

1. **Continuous measurement metrics** (primary for body cm)
2. **Classification metrics** after binning cm into size classes (confusion matrix, precision, recall, F1)

Reproduce:

```text
python compute_accuracy.py
```

---

### 6.1 Continuous formulas

```text
abs_error_cm = |digital − truth|
pct_error    = abs_error_cm / truth × 100%
accuracy%    = 100% − pct_error
MAE          = mean(abs_error_cm) over subjects
```

### 6.2 Continuous results — `measured` subset (subjects 0, 1, 11, 12, 16)

| Metric | MAE (cm) | Mean accuracy% |
|--------|----------|----------------|
| Height | 7.6 | **95.3%** |
| Waist | 20.1 | **71.5%** |
| Chest | 65.4 | **30.0%** |
| Overall (mean of three) | — | **65.6%** |

### 6.3 Continuous results — all 9 subjects (includes `tbr`)

| Metric | MAE (cm) | Mean accuracy% |
|--------|----------|----------------|
| Height | 6.8 | **95.8%** |
| Waist | 14.8 | **79.8%** |
| Chest | 56.7 | **41.7%** |
| Overall (mean of three) | — | **72.4%** |

Prefer the **measured** table (6.2) for the official write-up score; all-subjects is supporting context.

---

### 6.4 Classification setup (bins)

Continuous cm → discrete classes so we can show confusion matrix / F1:

| Metric | Class bins |
|--------|------------|
| Height | short &lt;165 · medium 165–175 · tall ≥175 |
| Chest | small &lt;95 · medium 95–110 · large ≥110 |
| Waist | small &lt;75 · medium 75–95 · large ≥95 |

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 × precision × recall / (precision + recall)
F1 macro  = unweighted mean of per-class F1
F1 weighted = support-weighted mean of per-class F1
```

Confusion matrix: **rows = truth (tape/label class)**, **cols = predicted (digital class)**.

---

### 6.5 Classification results — `measured` subset (n=5)

#### Height

| sid | truth_cm | pred_cm | true_cls | pred_cls |
|-----|----------|---------|----------|----------|
| 0 | 159.0 | 174.8 | short | medium |
| 1 | 157.0 | 166.3 | short | medium |
| 11 | 163.0 | 169.6 | short | medium |
| 12 | 165.0 | 167.9 | medium | medium |
| 16 | 175.0 | 171.6 | tall | medium |

Confusion matrix (labels: short, medium, tall):

```text
[[0 3 0]
 [0 1 0]
 [0 1 0]]
```

| Score | Value |
|-------|-------|
| Classification accuracy | 20.0% (1/5) |
| F1 macro | 0.111 |
| F1 weighted | 0.067 |

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| short | 0.000 | 0.000 | 0.000 | 3 |
| medium | 0.200 | 1.000 | 0.333 | 1 |
| tall | 0.000 | 0.000 | 0.000 | 1 |

#### Chest

| sid | truth_cm | pred_cm | true_cls | pred_cls |
|-----|----------|---------|----------|----------|
| 0 | 79.0 | 105.3 | small | medium |
| 1 | 93.0 | 186.6 | small | large |
| 11 | 86.0 | 104.8 | small | medium |
| 12 | 90.0 | 192.8 | small | large |
| 16 | 107.0 | 192.6 | medium | large |

Confusion matrix (labels: small, medium, large):

```text
[[0 2 2]
 [0 0 1]
 [0 0 0]]
```

| Score | Value |
|-------|-------|
| Classification accuracy | 0.0% |
| F1 macro | 0.000 |
| F1 weighted | 0.000 |

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| small | 0.000 | 0.000 | 0.000 | 4 |
| medium | 0.000 | 0.000 | 0.000 | 1 |
| large | 0.000 | 0.000 | 0.000 | 0 |

#### Waist

| sid | truth_cm | pred_cm | true_cls | pred_cls |
|-----|----------|---------|----------|----------|
| 0 | 68.0 | 94.5 | small | medium |
| 1 | 79.0 | 95.0 | medium | large |
| 11 | 65.0 | 96.8 | small | large |
| 12 | 73.0 | 94.3 | small | medium |
| 16 | 92.0 | 96.7 | medium | large |

Confusion matrix (labels: small, medium, large):

```text
[[0 2 1]
 [0 0 2]
 [0 0 0]]
```

| Score | Value |
|-------|-------|
| Classification accuracy | 0.0% |
| F1 macro | 0.000 |
| F1 weighted | 0.000 |

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| small | 0.000 | 0.000 | 0.000 | 3 |
| medium | 0.000 | 0.000 | 0.000 | 2 |
| large | 0.000 | 0.000 | 0.000 | 0 |

---

### 6.6 Classification results — all subjects (n=9)

| Metric | Acc | F1 macro | F1 weighted | Confusion matrix (row=truth) |
|--------|-----|----------|-------------|------------------------------|
| Height | 33.3% | 0.167 | 0.167 | `[[0 5 0],[0 3 0],[0 1 0]]` (short/med/tall) |
| Chest | 11.1% | 0.095 | 0.032 | `[[0 3 2],[0 0 3],[0 0 1]]` (small/med/large) |
| Waist | 22.2% | 0.179 | 0.175 | `[[0 2 1],[0 1 3],[0 1 1]]` (small/med/large) |

Height all-subjects per-class F1: short 0.000 · medium 0.500 · tall 0.000  
Chest all-subjects per-class F1: small 0.000 · medium 0.000 · large 0.286  
Waist all-subjects per-class F1: small 0.000 · medium 0.250 · large 0.286  

---

### 6.7 How to present this honestly

- **Primary score = continuous MAE / accuracy%.** Height ~95%, waist ~72%, chest ~30%, overall ~66%.
- **Confusion matrix / F1 are secondary.** They look much worse because small cm errors can cross bin boundaries, and chest/waist digital values are often oversized (arms in chest slice).
- Do **not** claim “the model is 65% accurate at everything.” Always give the per-metric continuous breakdown, then optionally show binned F1.

---

## 7. Why chest is hard (and other error sources)

### Main chest failure mode (method)
We measure chest by slicing the **digital T-pose** near chest height and wrapping around the outline. In T-pose, **arms stick out near that height**, so the ring can include arm+torso and report huge values (e.g. 180–190 cm vs truth ~90 cm). Waist is lower, so arms usually do not enter that slice as badly.

### Other real-world factors
- **Camera angle:** many phones/low angles (camera near ground looking up) distort proportions.
- **Clothing:** hides true body surface.
- **Single image:** monocular ambiguity (no lab multi-view scanner).
- **Dataset size:** only 9 subjects here; scores are indicative, not a large official benchmark.
- **Label quality:** `tbr` rows are weaker than `measured`.
- **Fixed % landmarks:** 0.74 / 0.58 are heuristics, not true anatomical markers.

Selfie images are usually poor for this task (cropped body). Side images can help body depth, but front full-body was our primary input. One image can produce all three measurements from one shape; using different views per metric is optional future work.

---

## 8. Troubleshooting — what broke and how we fixed it

| Problem | Cause | Fix |
|---------|--------|-----|
| `conda` not found in Cursor PowerShell | Conda not on PATH | `conda init powershell` / Anaconda Prompt |
| `'cp' is not recognized` | Linux copy command on Windows | Patched `main/config.py` to use `shutil` |
| `FormatCode(... verify=...)` | yapf / mmcv clash | Avoided broken `cfg.dump()` path |
| `cannot import name 'bool' from numpy` | NumPy too new for `chumpy` | `pip install numpy==1.23.5` |
| `Unable to load EGL library` | Linux OpenGL (EGL) on Windows | Patched `vis.py`; vertex overlay |
| mmdet config has no `model` | Config file was empty (0 bytes) | Restored Faster R-CNN config |
| `torchgeometry` bool subtraction | Old library vs newer PyTorch | Cast masks to float |
| Wrong image names | Needs `000001.jpg` style | Renamed / copied with 6-digit names |
| SMPL files in wrong folder | Placed under `smplx/` | Moved to `human_model_files/smpl/` |
| SHAPY Model Agency URLs | Many links expired | Switched to local labeled benchmark images |

---

## 9. Validation plan (also usable with a physical tape)

We validated **without a lab-grade 3D scanner** by using labeled real photos (benchmark). The same plan works with a cloth measuring tape:

1. Same person for photos + ground truth
2. Ground truth = tape **or** trusted dataset labels (height, chest, waist)
3. Run pipeline → digital cm
4. Compare with continuous metrics (MAE, % error, accuracy%) and optional binned classification metrics (confusion matrix, precision, recall, F1)
5. Prefer consistent photo protocol:
   - Full body, standing, arms slightly away
   - Camera near eye/chest level (not ground-looking-up)
   - Front (+ optional side)
   - Fitted clothes if possible

Optional future improvements:
- Better chest measurement (torso-only landmarks; exclude arms)
- Fuse front + side predictions
- Larger labeled set

---

## 10. What we learned

1. Clone gets **code**; model weights and body templates are separate downloads.
2. SMPLer-X + SMPL-X work together (AI + doll).
3. Height/chest/waist are the required three metrics; the same mesh can support more (e.g. hips) later.
4. Accuracy needs ground truth; model outputs alone are not enough.
5. On this benchmark: height good, waist moderate, chest limited by measurement method + photo conditions.
6. Windows setup failures are expected; documenting them is part of the work.

---

## 11. How to reproduce (checklist)

1. `conda activate smplerx`
2. Confirm models under `pretrained_models/` and `human_model_files/`
3. Put front images in `demo/images/my_person/` as `000001.jpg`…
4. Run inference from `main/` (`--start 1 --end N`)
5. `python measure_body.py`
6. `python compute_accuracy.py` (if using the benchmark mapping)

### UI (upload photo in browser)

```text
conda activate smplerx
cd "C:\Users\Asus\Desktop\demo ml\SMPLer-X"
python app_ui.py
```

Open http://127.0.0.1:7860 — upload a standing photo, click **Measure body**.  
Optional tape fields show accuracy%. First run can take a few minutes while models load.

---

## 12. References

- SMPLer-X: https://github.com/MotrixLab/SMPLer-X
- SMPL-X: https://smpl-x.is.tue.mpg.de/
- SMPL: https://smpl.is.tue.mpg.de/
- SHAPY datasets (inspected): https://shapy.is.tue.mpg.de/
- Body measurements sample source: https://huggingface.co/datasets/UniqueData/body-measurements-dataset
- Assignment: Option A — Body Measurement Mini-Pipeline (pretrained SMPL-X-based model, ≥3 measurements, write-up, validation without lab scanner)
