#!/usr/bin/env python3

from __future__ import annotations
from pathlib import Path
import subprocess, sys, shutil, yaml, random
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont

YOLOV5_DIR = Path(r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\yolov5")
DATA_YAML  = Path(r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\dataset\dataset_yolo_tiled\data.yaml")
OUT_ROOT   = Path(r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\artifacts_student\full_eval_valpy")


MODELS = {
    "teacher_baseline":          r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\runs\train\exp110\weights\best.pt",
    "student_fp_filter_only":    r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\artifacts_student\yolov5_runs\student_fp_filter_only\weights\best.pt",
    "student_refine_only":       r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\artifacts_student\yolov5_runs\student_refine_only\weights\best.pt",
    "student_teacher_only":      r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\artifacts_student\yolov5_runs\student_teacher_only\weights\best.pt",
    "student_fp_filter_plus_refine": r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\artifacts_student\yolov5_runs\student_fp_filter_plus_refine\weights\best.pt",
}


IMG_SIZE    = 1536
BATCH       = 8                
DEVICE      = "0" if shutil.which("nvidia-smi") else "cpu"
SAVE_TXT    = True              
SAVE_CONF   = True
SAVE_IMAGES = False             


SAMPLE_N        = 25
RANDOM_SAMPLE   = True          
LINE_W          = 3
TEXT_PAD        = 2


COMPUTE_CUSTOM_METRICS = False  


def must_exist(p: Path, desc: str):
    if not Path(p).exists():
        raise FileNotFoundError(f"Missing {desc}: {p}")

def load_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))

def get_images_root(d: dict, yaml_path: Path) -> Path:
    base = Path(d.get("path", yaml_path.parent)).resolve()
    if "test" not in d or not d["test"]:
        raise RuntimeError(f"Your data.yaml must define a 'test' split. Got keys: {list(d.keys())}")
    root = Path(d["test"])
    if not root.is_absolute():
        root = (base / root).resolve()
    return root

def infer_labels_root(images_root: Path) -> Path:
    
    parts = list(images_root.parts)
    if "images" in parts:
        i = parts.index("images")
        labels_root = Path(*parts[:i], "labels", *parts[i+1:])
    else:
        
        labels_root = images_root.parent.parent / "labels" / images_root.name
    return labels_root

def list_images(root: Path) -> List[Path]:
    exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp",
            ".JPG",".JPEG",".PNG",".BMP",".TIF",".TIFF",".WEBP"}
    return sorted([p for p in root.rglob("*") if p.suffix in exts])

def img_to_label_path(img: Path, images_root: Path, labels_root: Path) -> Path:
    rel = img.relative_to(images_root)
    return (labels_root / rel).with_suffix(".txt")

def xywhn_to_xyxy(xc, yc, w, h, W, H):
    x1 = (xc - w/2.0) * W
    y1 = (yc - h/2.0) * H
    x2 = (xc + w/2.0) * W
    y2 = (yc + h/2.0) * H
    return x1, y1, x2, y2

def clamp_xyxy(x1,y1,x2,y2,W,H):
    x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W-1))
    y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H-1))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1,y1,x2,y2

def pick_color(cid: int) -> str:
    palette = [
        "lime", "red", "deepskyblue", "orange", "magenta", "yellow",
        "cyan", "gold", "violet", "springgreen", "dodgerblue", "salmon"
    ]
    return palette[cid % len(palette)]

def measure_text(draw: ImageDraw.ImageDraw, text: str, font: Optional[ImageFont.ImageFont]) -> Tuple[int, int]:
    """Return (width, height) for text, Pillow ≥10 and older."""
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        pass
    if font is not None:
        try:
            bbox = font.getbbox(text)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            pass
    try:
        return draw.textsize(text, font=font)  
    except Exception:
        pass
    return 7 * len(text), 11

def draw_overlay(img_path: Path, pred_path: Path, out_path: Path, id2name: Dict[int,str]):
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    if pred_path.exists():
        for line in pred_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            try:
                cid = int(float(parts[0]))
                xc, yc, w, h, conf = map(float, parts[1:6])
            except Exception:
                continue
            x1,y1,x2,y2 = xywhn_to_xyxy(xc,yc,w,h,W,H)
            x1,y1,x2,y2 = clamp_xyxy(x1,y1,x2,y2,W,H)
            color = pick_color(cid)
            draw.rectangle([x1,y1,x2,y2], outline=color, width=LINE_W)
            label = f"{id2name.get(cid, str(cid))} {conf:.2f}"
            if font is None:
                draw.text((x1+2, max(0,y1-12)), label, fill=color)
            else:
                tw, th = measure_text(draw, label, font)
                yb = max(0, y1 - 2)
                yt = max(0, yb - th - 2*TEXT_PAD)
                if yb < yt:
                    yb = yt
                from PIL import ImageColor
                
                draw.rectangle([x1, yt, x1 + tw + 2*TEXT_PAD, yb], fill=color)
                draw.text((x1 + TEXT_PAD, yt + TEXT_PAD), label, fill="black", font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)

def build_id2name(yaml_path: Path) -> Dict[int,str]:
    y = load_yaml(yaml_path)
    names = y.get("names")
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    elif isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    else:
        raise ValueError("data.yaml 'names' must be list or dict")


def run_val(weights: Path, data_yaml: Path, save_root: Path) -> Path:
    
    val_py = YOLOV5_DIR / "val.py"
    must_exist(val_py, "YOLOv5 val.py")
    out_dir = save_root / "val"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(val_py),
        "--weights",   str(weights),
        "--data",      str(data_yaml),
        "--task",      "test",                      
        "--imgsz",     str(IMG_SIZE),
        "--batch",     str(BATCH),
        "--device",    str(DEVICE),
        "--project",   str(out_dir),
        "--name",      "exp",
        "--exist-ok",
        "--verbose",
    ]
    if SAVE_TXT:    cmd += ["--save-txt"]
    if SAVE_CONF:   cmd += ["--save-conf"]
    if SAVE_IMAGES: cmd += ["--save"]

    proc = subprocess.run(cmd, cwd=str(YOLOV5_DIR), capture_output=True, text=True)
    (save_root / "val_stdout.txt").write_text(proc.stdout + "\n\n--- STDERR ---\n" + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"val.py failed for {weights}. See {save_root/'val_stdout.txt'}")

    
    return out_dir / "exp"


def maybe_compute_custom_metrics(*args, **kwargs):
    pass

# ====================== main ======================
if __name__ == "__main__":
    random.seed(0)

    must_exist(YOLOV5_DIR, "YOLOv5 repo dir")
    must_exist(YOLOV5_DIR / "val.py", "YOLOv5 val.py")
    must_exist(DATA_YAML, "dataset YAML")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    data = load_yaml(DATA_YAML)
    images_root = get_images_root(data, DATA_YAML)
    labels_root = infer_labels_root(images_root)
    must_exist(images_root, "images root (test)")
    must_exist(labels_root, "labels root (test)")

    id2name = build_id2name(DATA_YAML)
    nc = len(id2name)

    all_images = list_images(images_root)
    if not all_images:
        raise RuntimeError(f"No images found under {images_root}")
    print(f"Found {len(all_images)} images in test split.")

    for model_name, weights in MODELS.items():
        print(f"\n=== [{model_name}] ===")
        weights_path = Path(weights)
        must_exist(weights_path, f"weights for {model_name}")
        save_root = OUT_ROOT / model_name
        save_root.mkdir(parents=True, exist_ok=True)

        print("Running val.py --task test …")
        val_run_dir = run_val(weights_path, DATA_YAML, save_root)
        print(f"YOLOv5 artifacts:", val_run_dir)

        labels_pred_dir = val_run_dir / "labels"
        if not labels_pred_dir.exists():
            print("Warning: val.py did not produce labels/*.txt (enable --save-txt). Skipping overlays.")
        else:
            print(f"Rendering overlay SAMPLE ({SAMPLE_N}) …")
            overlays_sample = save_root / "overlays" / "sample25"
            if RANDOM_SAMPLE and len(all_images) > SAMPLE_N:
                subset = random.sample(all_images, SAMPLE_N)
            else:
                subset = all_images[:SAMPLE_N]
            done = 0
            for img in subset:
                out_img = overlays_sample / (img.stem + ".jpg")
                pred_txt = labels_pred_dir / (img.stem + ".txt")
                draw_overlay(img, pred_txt, out_img, id2name)
                done += 1
                if done % 5 == 0 or done == len(subset):
                    print(f"  overlays: {done}/{len(subset)}")

        
        if COMPUTE_CUSTOM_METRICS:
            maybe_compute_custom_metrics()

        print(f"Done → {save_root}")

    print("\nAll evaluations complete. Artifacts under:", OUT_ROOT)
