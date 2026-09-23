"""
Body Measurement Mini-Pipeline — Gradio UI

Upload a photo → run SMPLer-X inference → measure height / chest / waist.
Optionally enter tape values to see accuracy%.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# Local measurement helpers
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "common" / "utils"))

from measure_body import build_tpose_vertices, measure_vertices  # noqa: E402

try:
    import gradio as gr
except ImportError:
    raise SystemExit(
        "Gradio is not installed. Run:\n"
        "  conda activate smplerx\n"
        "  pip install 'gradio==3.50.2'\n"
    )

try:
    import smplx
except ImportError:
    smplx = None

UPLOAD_DIR = ROOT / "demo" / "images" / "ui_upload"
RESULT_DIR = ROOT / "demo" / "results" / "ui_upload"
MAIN_DIR = ROOT / "main"
MODEL_PATH = ROOT / "common" / "utils" / "human_model_files"
PRETRAINED = "smpler_x_s32"

_smplx_model = None


def get_smplx_model():
    global _smplx_model
    if _smplx_model is None:
        if smplx is None:
            raise RuntimeError("smplx package missing — activate the smplerx conda env.")
        _smplx_model = smplx.create(
            str(MODEL_PATH),
            model_type="smplx",
            gender="neutral",
            use_face_contour=False,
            num_betas=10,
            num_expression_coeffs=10,
            use_pca=False,
            flat_hand_mean=True,
        )
        _smplx_model.eval()
    return _smplx_model


def continuous_metrics(pred: float, truth: float):
    err = abs(pred - truth)
    pct = (err / truth * 100.0) if truth else float("nan")
    acc = 100.0 - pct
    return err, pct, acc


def save_upload_as_frame(image) -> Path:
    """Save Gradio image (path / numpy / PIL) as 000001.jpg for inference."""
    from PIL import Image

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    # clear previous uploads so frame 1 is this image only
    for old in UPLOAD_DIR.glob("*"):
        if old.is_file():
            old.unlink()

    out = UPLOAD_DIR / "000001.jpg"

    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image.astype("uint8")).convert("RGB")
    else:
        img = image.convert("RGB")

    img.save(out, format="JPEG", quality=95)
    return out


def run_inference() -> None:
    """Call SMPLer-X inference.py the same way as the write-up checklist."""
    if RESULT_DIR.exists():
        shutil.rmtree(RESULT_DIR)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "inference.py",
        "--num_gpus",
        "1",
        "--exp_name",
        "output/demo_ui",
        "--pretrained_model",
        PRETRAINED,
        "--agora_benchmark",
        "agora_model",
        "--img_path",
        str(UPLOAD_DIR),
        "--start",
        "1",
        "--end",
        "1",
        "--output_folder",
        str(RESULT_DIR),
        "--show_verts",
        "--show_bbox",
        "--save_mesh",
    ]
    env = os.environ.copy()
    # Keep CUDA visible if present
    proc = subprocess.run(
        cmd,
        cwd=str(MAIN_DIR),
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-2500:]
        raise RuntimeError(f"Inference failed (exit {proc.returncode}).\n\n{tail}")


def measure_from_results():
    smplx_dir = RESULT_DIR / "smplx"
    npz_files = sorted(smplx_dir.glob("*.npz"))
    if not npz_files:
        raise RuntimeError(
            f"No .npz outputs in {smplx_dir}. Inference may have failed to detect a person."
        )

    model = get_smplx_model()
    data = np.load(npz_files[0])
    betas = data["betas"]
    verts, joints = build_tpose_vertices(betas, model, return_joints=True)
    return measure_vertices(verts, joints), npz_files[0].name


def find_overlay_image():
    """Prefer the rendered overlay written by inference to results/.../img/."""
    img_dir = RESULT_DIR / "img"
    preferred = img_dir / "000001.jpg"
    if preferred.is_file():
        return str(preferred)
    if img_dir.is_dir():
        for p in sorted(img_dir.glob("*")):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                return str(p)
    return None


HEIGHT_BINS = [("short", 0, 165), ("medium", 165, 175), ("tall", 175, 300)]
CHEST_BINS = [("small", 0, 95), ("medium", 95, 105), ("large", 105, 200)]
WAIST_BINS = [("small", 0, 80), ("medium", 80, 95), ("large", 95, 200)]
BIN_MAP = {
    "height_cm": HEIGHT_BINS,
    "chest_cm": CHEST_BINS,
    "waist_cm": WAIST_BINS,
}


def bin_value(value, bins):
    for name, lo, hi in bins:
        if lo <= value < hi:
            return name
    return bins[-1][0]


def format_results(m: dict, npz_name: str, truth: dict) -> str:
    lines = [
        "## Digital measurements",
        f"- Source shape file: `{npz_name}`",
        "",
        "| Metric | Digital (cm) |",
        "|--------|-------------:|",
        f"| Height | {m['height_cm']:.1f} |",
        f"| Chest  | {m['chest_cm']:.1f} |",
        f"| Waist  | {m['waist_cm']:.1f} |",
        "",
    ]

    # Ignore 0 / empty — Gradio Number often defaults to 0 when unused
    filled = {
        k: truth[k]
        for k in ("height_cm", "chest_cm", "waist_cm")
        if truth.get(k) is not None and float(truth[k]) > 0
    }
    if filled:
        rows = []
        for key, label in (
            ("height_cm", "Height"),
            ("chest_cm", "Chest"),
            ("waist_cm", "Waist"),
        ):
            if key not in filled:
                continue
            t = float(filled[key])
            p = float(m[key])
            err, pct, acc = continuous_metrics(p, t)
            rows.append(
                {
                    "label": label,
                    "key": key,
                    "truth": t,
                    "digital": p,
                    "abs_err": err,
                    "pct_err": pct,
                    "acc": acc,
                }
            )

        lines += [
            "## Accuracy vs your tape / labels",
            "",
            "`accuracy% = 100 − |digital − truth| / truth × 100`",
            "",
            "| Metric | Truth (cm) | Digital (cm) | Abs err (cm) | % err | Acc% |",
            "|--------|-----------:|-------------:|-------------:|------:|-----:|",
        ]
        for r in rows:
            lines.append(
                f"| {r['label']} | {r['truth']:.1f} | {r['digital']:.1f} | "
                f"{r['abs_err']:.1f} | {r['pct_err']:.1f}% | **{r['acc']:.1f}%** |"
            )

        mae = float(np.mean([r["abs_err"] for r in rows]))
        mean_pct = float(np.mean([r["pct_err"] for r in rows]))
        overall = float(np.mean([r["acc"] for r in rows]))
        lines += [
            "",
            "### Summary metrics",
            "",
            "| Summary | Value |",
            "|---------|------:|",
            f"| MAE (mean abs error) | {mae:.1f} cm |",
            f"| Mean percent error | {mean_pct:.1f}% |",
            f"| **Overall mean accuracy** | **{overall:.1f}%** |",
            "",
            "### Size class (same bins as the CLI report)",
            "",
            "| Metric | Tape class | Digital class | Match |",
            "|--------|------------|---------------|-------|",
        ]
        for r in rows:
            t_cls = bin_value(r["truth"], BIN_MAP[r["key"]])
            p_cls = bin_value(r["digital"], BIN_MAP[r["key"]])
            match = "yes" if t_cls == p_cls else "no"
            lines.append(f"| {r['label']} | {t_cls} | {p_cls} | {match} |")
    else:
        lines += [
            "_No tape values entered._ Fill **Truth height / chest / waist** above, "
            "then click **Measure body** again to see Acc%, MAE, and class comparison.",
        ]

    lines += [
        "",
        "### Notes",
        "- Chest/waist use a torso-only slice (arms in T-pose are ignored).",
        "- Measurements use shape betas → digital T-pose → geometry (not a separate ML measurer).",
    ]
    return "\n".join(lines)


def _as_optional_float(v):
    if v is None or v == "":
        return None
    try:
        val = float(v)
    except (TypeError, ValueError):
        return None
    if val == 0:
        return None
    return val


def process(image, truth_height, truth_chest, truth_waist):
    if image is None:
        yield None, "Please upload a full-body standing photo first.", "Waiting for a photo."
        return

    try:
        yield None, "Saving upload…", "Saving photo…"
        save_upload_as_frame(image)

        yield None, "Running SMPLer-X (this can take 1–3 minutes)…", "Detecting person and estimating shape…"
        run_inference()

        yield None, "Measuring height / chest / waist…", "Measuring mesh…"
        m, npz_name = measure_from_results()

        truth = {
            "height_cm": _as_optional_float(truth_height),
            "chest_cm": _as_optional_float(truth_chest),
            "waist_cm": _as_optional_float(truth_waist),
        }

        overlay = find_overlay_image()
        if overlay is None:
            overlay = str(UPLOAD_DIR / "000001.jpg")
        md = format_results(m, npz_name, truth)
        yield overlay, md, "Done."
    except Exception as e:
        yield None, f"**Error**\n\n```\n{e}\n```", "Failed."


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700&family=Space+Grotesk:wght@500;700&display=swap');

:root, .gradio-container, .dark, .gradio-container.dark {
  color-scheme: light !important;
  --body-background-fill: #f4f7f5 !important;
  --background-fill-primary: #ffffff !important;
  --background-fill-secondary: #ffffff !important;
  --block-background-fill: #ffffff !important;
  --input-background-fill: #ffffff !important;
  --panel-background-fill: #ffffff !important;
  --body-text-color: #14201b !important;
  --body-text-color-subdued: #3d4f46 !important;
  --block-title-text-color: #14201b !important;
  --block-label-text-color: #ffffff !important;
  --block-label-background-fill: #0f7a4a !important;
  --block-info-text-color: #3d4f46 !important;
  --input-text-color: #14201b !important;
  --neutral-800: #e8eee9 !important;
  --neutral-900: #ffffff !important;
  --neutral-950: #14201b !important;
}

.gradio-container {
  font-family: 'Manrope', sans-serif !important;
  color: #14201b !important;
  background: #f4f7f5 !important;
  max-width: 1100px !important;
}

/* Cards / blocks / accordion: light, never navy */
.gradio-container .block,
.gradio-container .form,
.gradio-container .panel,
.gradio-container .gr-box,
.gradio-container .gr-panel,
.gradio-container .gr-padded,
.gradio-container .wrap,
.gradio-container .contain,
.gradio-container .accordion,
.gradio-container .label-wrap,
.gradio-container .image-container,
.gradio-container .image-preview,
.gradio-container .upload-container,
.gradio-container .empty,
.gradio-container .or {
  background: #ffffff !important;
  color: #14201b !important;
  border-color: #c9d5ce !important;
}

.gradio-container .label-wrap span,
.gradio-container .label-wrap svg,
.gradio-container .accordion span,
.gradio-container .accordion svg {
  color: #14201b !important;
  fill: #14201b !important;
}

/* Image dropzone / preview canvas */
.gradio-container .image-container,
.gradio-container .image-container > div,
.gradio-container [data-testid="image"],
.gradio-container .image-preview,
.gradio-container .upload-container,
.gradio-container canvas,
.gradio-container .unpadded_box {
  background: #eef3f0 !important;
}

/* Component labels (green pills) — white text on green */
.gradio-container .block > .label-wrap,
.gradio-container span.float,
.gradio-container .block .label-wrap span {
  color: #14201b !important;
}

.gradio-container .block .label-wrap {
  background: transparent !important;
}

/* Inputs */
.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container .gr-input,
.gradio-container .gr-box input {
  color: #14201b !important;
  background: #ffffff !important;
  border: 1px solid #c9d5ce !important;
}

.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
  color: #5b6b63 !important;
  opacity: 1 !important;
}

.gradio-container .prose,
.gradio-container .markdown,
.gradio-container .md,
.gradio-container p,
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container li,
.gradio-container td,
.gradio-container th,
.gradio-container label {
  color: #14201b !important;
}

.gradio-container .prose code,
.gradio-container .markdown code {
  color: #0b3d28 !important;
  background: #e7f3ec !important;
}

#brand-title {
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 2.1rem !important;
  font-weight: 700 !important;
  letter-spacing: -0.03em !important;
  color: #14201b !important;
  margin-bottom: 0.15rem !important;
}

#brand-sub {
  color: #3d4f46 !important;
  font-size: 1.02rem !important;
  max-width: 42rem;
}

button.primary, .primary, button.primary span {
  background: #0f7a4a !important;
  color: #ffffff !important;
  border: none !important;
}

footer { display: none !important; }
.progress-bar, .meta-text, .eta-bar, .wrap.progress-level { display: none !important; }
"""


def build_ui():
    theme = gr.themes.Soft(
        primary_hue="green",
        secondary_hue="stone",
        neutral_hue="stone",
        text_size="lg",
    ).set(
        body_text_color="#14201b",
        body_text_color_subdued="#3d4f46",
        block_title_text_color="#14201b",
        block_label_text_color="#14201b",
        button_primary_text_color="#ffffff",
        button_secondary_text_color="#14201b",
        background_fill_primary="#f4f7f5",
        background_fill_secondary="#ffffff",
        block_background_fill="#ffffff",
        border_color_primary="#c9d5ce",
    )

    with gr.Blocks(css=CUSTOM_CSS, theme=theme, title="Body Measurement Mini-Pipeline") as demo:
        gr.HTML(
            """
            <div id="brand-title">MeasureFit</div>
            <p id="brand-sub">
              Upload a standing photo. SMPLer-X estimates body shape; we measure
              height, chest, and waist from the mesh. Optionally add tape values to score accuracy.
            </p>
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="pil", label="Standing photo (full body preferred)")
                with gr.Accordion(
                    "Optional tape / ground truth (cm) — enter real tape values to see Acc% and metrics",
                    open=True,
                ):
                    gr.Markdown(
                        "Leave blank if you only want digital cm. "
                        "If you type your tape measurements (not 0), the right panel shows "
                        "Truth vs Digital, abs error, % error, Acc%, MAE, and size class."
                    )
                    th = gr.Number(label="Truth height (cm)", precision=1, value=None)
                    tc = gr.Number(label="Truth chest (cm)", precision=1, value=None)
                    tw = gr.Number(label="Truth waist (cm)", precision=1, value=None)
                run_btn = gr.Button("Measure body", variant="primary")
                gr.Markdown(
                    "Tip: use a front full-body photo, camera near chest height. "
                    "First run can take 1–3 minutes while models load."
                )

            with gr.Column(scale=1):
                status = gr.Markdown("Ready — upload a photo and click **Measure body**.")
                overlay = gr.Image(label="Pipeline overlay (if available)", interactive=False)
                report = gr.Markdown("Results will appear here.")

        run_btn.click(
            fn=process,
            inputs=[image, th, tc, tw],
            outputs=[overlay, report, status],
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.queue(concurrency_count=1)
    ui.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, show_error=True)
