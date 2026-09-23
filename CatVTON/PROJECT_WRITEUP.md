# Option B — Fit-Rendering Mini-Pipeline (CatVTON)

**Goal:** Run a pretrained virtual try-on model (CatVTON) on a person photo + a garment photo, document setup issues, and assess output quality.

**Hardware used:** Windows 11, NVIDIA GeForce RTX 3050, Anaconda env `catvton` (Python 3.9)

**Base model license:** CatVTON materials (code / checkpoints / demo) are under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) — non-commercial use; give credit and share alike.

---

## What each part does (to get the result)

Big picture:

```text
Person photo + Garment photo
  → find body / clothes region (mask)
  → Stable Diffusion inpainting (CatVTON)
  → result: person appearing to wear that garment
```

### 1. Your inputs

| Input | Role |
|--------|------|
| **Person image** | Who to dress (pose, body, face, background) |
| **Garment image** | What to put on (shirt, dress, etc.) |
| **Cloth type** (upper / lower / overall) | Tells the masker which body area to replace |

### 2. AutoMasker (Detectron2 + DensePose + SCHP)

Before painting, the app builds a **mask** = “these pixels are the old clothes / try-on area.”

| Piece | Job |
|--------|-----|
| **Detectron2** | Finds the person in the photo |
| **DensePose** | Maps body surface (torso, arms, etc.) |
| **SCHP** | Human parsing — labels body parts / clothing regions more finely |

Together they decide: replace the clothing region, keep face / hands / background when possible.

Without this step, the model would not know **where** to put the garment.

### 3. Resize / crop / pad

Person and cloth images are resized to the model resolution (e.g. **768×1024**). That keeps the network happy and lines up person + garment.

### 4. CatVTON pipeline (the AI “painter”)

Built on **Diffusers** + **Stable Diffusion inpainting**.

| Piece | Job |
|--------|-----|
| **VAE** | Compresses images into a smaller latent space (faster than full pixels) |
| **UNet** | At each step, predicts how to turn noise into a realistic clothed person |
| **Scheduler (DDIM)** | Controls denoising steps (often **50**). Each step cleans the image a bit more |
| **CatVTON attention / weights** | Trained parts that link the garment image to the person |
| **Mask** | Limits painting to the clothing region so face/background stay |

Roughly: start from noise in the masked area → many denoising steps → look like that garment on that person.

On an RTX 3050, **50 steps** can take a long time (on the order of ~1 hour for a full run). That is expected for a small GPU.

### 5. Optional extras

| Piece | Job |
|--------|-----|
| **Safety checker** | Filters NSFW (not what draws the clothes) |
| **Repaint / composite** | Can blend result back with original background |
| **Gradio UI** | Upload buttons, progress bar, shows the final image |

### One-line summary

**DensePose/SCHP find where clothes go; CatVTON (SD inpainting + try-on weights) redraws that region so the person appears to wear the garment.**

---

## How to run (Windows — this machine)

### Prerequisites

- Anaconda / Miniconda
- NVIDIA GPU drivers (`nvidia-smi` works)
- Git
- **Microsoft C++ Build Tools** with **Desktop development with C++** (needed to compile Detectron2 on Windows)

### 1) Create env and clone

```powershell
conda create -n catvton python=3.9 -y
conda activate catvton
cd "C:\Users\Asus\Desktop\demo ml"
git clone https://github.com/Zheng-Chong/CatVTON.git
cd CatVTON
```

### 2) Install core packages (avoid broken pins)

Do **not** blindly run `pip install -r requirements.txt` on Windows — matplotlib / latest diffusers / Python 3.9 often fail.

```powershell
conda activate catvton
cd "C:\Users\Asus\Desktop\demo ml\CatVTON"

# GPU PyTorch (CUDA 11.8)
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118

# Diffusers for Python 3.9 (not git main — that needs Python 3.10+)
pip install "diffusers==0.30.3"

# Rest of stack (versions that worked on this PC)
pip install "numpy==1.26.4" "opencv-python==4.10.0.84" pillow PyYAML scipy scikit-image tqdm
pip install "transformers==4.46.3" fvcore cloudpickle omegaconf pycocotools av
pip install "gradio==4.41.0" "peft>=0.17.0" "accelerate>=1.2.0,<2.0" huggingface_hub
```

Check GPU:

```powershell
python -c "import torch; print(torch.__version__); print('cuda', torch.cuda.is_available())"
```

Expect something like `2.4.0+cu118` and `cuda True`.

### 3) Detectron2 + DensePose (for the Gradio auto-mask)

```powershell
conda activate catvton
cd "C:\Users\Asus\Desktop\demo ml\CatVTON"

# Short temp path avoids Windows path-length build failures
mkdir C:\tmp -ErrorAction SilentlyContinue
$env:TMP = "C:\tmp"
$env:TEMP = "C:\tmp"

git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
python -m pip install -e . --no-build-isolation

cd "C:\Users\Asus\Desktop\demo ml\CatVTON\detectron2\projects\DensePose"
pip install -e . --no-build-isolation

# DensePose may bump numpy to 2.x — pin back for Torch
pip install "numpy==1.26.4"
```

### 4) Start the Gradio app

```powershell
conda activate catvton
cd "C:\Users\Asus\Desktop\demo ml\CatVTON"
$env:CUDA_VISIBLE_DEVICES=0
python app.py --output_dir="resource/demo/output" --mixed_precision="fp16" --allow_tf32
```

- First run downloads CatVTON + SD inpainting weights from Hugging Face (can take a while).
- Open **http://127.0.0.1:7860** (prefer Chrome over the IDE browser).
- Upload **person** + **garment**, pick cloth type, generate.
- Leave the terminal open until progress finishes (e.g. `50/50`).

If port **7860** is busy (MeasureFit / old Gradio still running):

```powershell
Get-NetTCPConnection -LocalPort 7860 | Select-Object OwningProcess
Stop-Process -Id <PID> -Force
```

### 5) Gradio “Internal Server Error” fix (this PC)

Newer FastAPI can send boolean JSON Schema values that crash Gradio 4.41’s API info parser (`Cannot parse schema True`).

If the page shows **Internal Server Error** but the terminal already says `Running on local URL`, patch:

`C:\Users\Asus\anaconda3\envs\catvton\lib\site-packages\gradio_client\utils.py`

- In `get_type`: if schema is not a `dict`, return `"any"`.
- In `_json_schema_to_python_type` / `json_schema_to_python_type`: if schema is `True` / `False` / not a dict, return `"Any"`.

Then restart `app.py`.

---

## Setup issues we hit (honest notes)

| Issue | What happened | Fix |
|--------|----------------|-----|
| `matplotlib==3.9.1` build | Tried to compile from source; freetype download failed | Install matplotlib via conda / prebuilt wheel |
| Diffusers from git | Needs Python **≥ 3.10** | Use `diffusers==0.30.3` on Python 3.9 |
| Torch `+cpu` | No GPU | Reinstall cu118 wheels |
| Detectron2 | Needs MSVC C++ Build Tools on Windows | Install “Desktop development with C++”, reboot |
| DensePose long path | `cl.exe` C1083 invalid argument | Build with `$env:TMP=C:\tmp` from local `projects/DensePose` |
| `clear_device_cache` | peft vs old accelerate | Upgrade accelerate to ≥ 1.2 |
| Torch 2.6 required for `.bin` | New transformers vs Torch 2.4 | Pin `transformers==4.46.3` |
| Gradio Internal Server Error | Schema `True` bug | Patch `gradio_client/utils.py` |
| Slow inference | 50 steps on RTX 3050 | Wait; or lower steps/resolution next time |

Backup if local UI fails: [Hugging Face CatVTON Space](https://huggingface.co/spaces/zhengchong/CatVTON)

---

## Output quality (what to write honestly)

Virtual try-on usually looks best with:

- Front person photo, clear lighting
- Flat / product garment shot
- Matching cloth type (upper vs overall)

It often fails or looks weak on:

- Logos / fine text
- Strong stripes / complex patterns
- Hands overlapping clothes
- Pose mismatch vs garment mannequin
- Long dresses, messy backgrounds

Treat the result as a **demo**, not production storefront quality, especially on a laptop GPU.

---

## Fine-tune plan (50–100 proprietary garments, rented 24–48 GB GPU) — outline

*(Writing plan only; we did not train here.)*

1. **Data:** 50–100 brand garment photos + paired person / mask pairs (or DressCode / VITON-style layout). Clean backgrounds help.
2. **Hardware:** Single rented GPU with **24–48 GB** VRAM (e.g. A5000 / A6000 / L40 class).
3. **Method:** Parameter-efficient fine-tune of CatVTON (repo cites ~49M trainable params) with LoRA / PEFT rather than full UNet training.
4. **Schedule:** Short run (hours, not weeks): freeze most of SD backbone; train attention / try-on adapters on brand clothes; validate on held-out pairs.
5. **License:** Base CatVTON is **CC BY-NC-SA 4.0** — commercial product use needs legal review; fine-tune for internal demos is usually OK if terms are followed.

---

## References

- CatVTON GitHub: https://github.com/Zheng-Chong/CatVTON
- Weights: https://huggingface.co/zhengchong/CatVTON
- Paper: https://arxiv.org/abs/2407.15886
- Online Space: https://huggingface.co/spaces/zhengchong/CatVTON