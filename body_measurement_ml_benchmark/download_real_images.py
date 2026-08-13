from pathlib import Path
import requests

BASE = "https://huggingface.co/datasets/UniqueData/body-measurements-dataset/resolve/main/"
SUBJECT_IDS = [0, 1, 10, 11, 12, 13, 14, 15, 16]

out = Path("images")
out.mkdir(exist_ok=True)

for sid in SUBJECT_IDS:
    subject_dir = out / str(sid)
    subject_dir.mkdir(exist_ok=True)
    for name in ["front_img.jpg", "side_img.jpg", "selfie_img.jpg", "measurements.json"]:
        url = f"{BASE}files/{sid}/{name}"
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        (subject_dir / name).write_bytes(r.content)
        print("downloaded", subject_dir / name)
