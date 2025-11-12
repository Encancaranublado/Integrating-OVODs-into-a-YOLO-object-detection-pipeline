#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import shutil, os, yaml


DATASET_ROOT = Path(r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\dataset\dataset_yolo_tiled")
IMAGES_TRAIN = DATASET_ROOT / "images" / "train"
IMAGES_VAL   = DATASET_ROOT / "images" / "val"
LABELS_VAL   = DATASET_ROOT / "labels" / "val"
DATA_YAML_IN = DATASET_ROOT / "data.yaml"


VARIANTS = [
    # ("teacher_only",
    # Path(r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\artifacts_student\yolo_train\labels")),
    ("rsclip_filtered",
     Path(r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\artifacts_student\rsclip_verified_train_fast\labels_verified")),
    ("owlv2_refined",
     Path(r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\artifacts_student\owlv2_visualgrounding_artifacts\owlv2_grounded_train\labels_grounded")),
    ("rsclip_owlv2",
     Path(r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\artifacts_student\rsclip_owlv2_refined\labels")),
]


OUT_ROOT_BASE = Path(r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\artifacts_student\student_datasets")

ALLOWED_IMG_EXT = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp",".JPG",".PNG",".TIF",".WEBP"}

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def copy_or_hardlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists(): return
    try:
        os.link(src, dst)  
    except Exception:
        shutil.copy2(src, dst)

def list_images(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.suffix in ALLOWED_IMG_EXT]

def normalize_to_5col(src_lbl: Path, dst_lbl: Path):
    """Write YOLO 5-col (cls xc yc w h); if 6th token exists, drop it."""
    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
    if not src_lbl.exists():
        dst_lbl.write_text("", encoding="utf-8")
        return
    out_lines = []
    for ln in src_lbl.read_text(encoding="utf-8").splitlines():
        parts = ln.strip().split()
        if len(parts) >= 5:
            out_lines.append(" ".join(parts[:5]))
    dst_lbl.write_text("\n".join(out_lines), encoding="utf-8")

def load_names(yaml_path: Path) -> tuple[int, list[str]]:
    y = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = y.get("names")
    if isinstance(names, dict):
        nc = len(names); ordered = [names[i] for i in range(nc)]
        return nc, ordered
    elif isinstance(names, list):
        return len(names), names
    else:
        raise ValueError("data.yaml 'names' must be list or dict")

def build_variant(variant_name: str, labels_dir: Path):
    if not labels_dir.exists():
        print(f"[SKIP] {variant_name}: labels dir missing -> {labels_dir}")
        return
    out_root   = OUT_ROOT_BASE / variant_name
    out_img_tr = ensure_dir(out_root / "images" / "train")
    out_lbl_tr = ensure_dir(out_root / "labels" / "train")
    data_yaml  = out_root / f"data_{variant_name}.yaml"

    # mirror train images
    imgs = list_images(IMAGES_TRAIN)
    print(f"[{variant_name}] linking {len(imgs)} train images …")
    for src in imgs:
        copy_or_hardlink(src, out_img_tr / src.name)

    # normalize training labels by stem
    print(f"[{variant_name}] copying/normalizing labels from {labels_dir} …")
    with_lbl = empty_lbl = 0
    for dst_img in out_img_tr.iterdir():
        if not dst_img.is_file(): continue
        stem = dst_img.stem
        src_lbl = labels_dir / f"{stem}.txt"
        dst_lbl = out_lbl_tr / f"{stem}.txt"
        if src_lbl.exists():
            normalize_to_5col(src_lbl, dst_lbl)
            with_lbl += 1
        else:
            dst_lbl.write_text("", encoding="utf-8")
            empty_lbl += 1
    print(f"[{variant_name}] labels: present={with_lbl} empty={empty_lbl}")

    # write data.yaml
    nc, names = load_names(DATA_YAML_IN)
    data = {
        "path": out_root.as_posix(),
        "train": "images/train",
        "val": IMAGES_VAL.as_posix(),  
        "nc": nc,
        "names": names,
    }
    data_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print(f"[{variant_name}] wrote {data_yaml}")

    
    runs_root = (OUT_ROOT_BASE / "yolov5_runs").as_posix()
    print(rf"""
# YOLOv5 training for {variant_name}
python "{(DATASET_ROOT.parent / 'yolov5' / 'train.py').as_posix()}" ^
  --img 1024 --batch 2 --epochs 25 ^
  --data "{data_yaml.as_posix()}" ^
  --weights yolov5m.pt ^
  --project "{runs_root}" --name "student_{variant_name}" --exist-ok ^
  --device 0 --workers 2 --save-period 5 --optimizer SGD --cos-lr
# best.pt / last.pt will be in: {runs_root}\student_{variant_name}\weights
""")

def main():
    
    for p in (DATASET_ROOT, IMAGES_TRAIN, IMAGES_VAL, LABELS_VAL, DATA_YAML_IN):
        if not p.exists(): raise FileNotFoundError(p)
    OUT_ROOT_BASE.mkdir(parents=True, exist_ok=True)
    for name, lbls in VARIANTS:
        build_variant(name, lbls)
    print("\nAll variants prepared.")

if __name__ == "__main__":
    main()
