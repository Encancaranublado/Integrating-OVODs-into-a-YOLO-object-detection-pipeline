#!/usr/bin/env python3


from __future__ import annotations

import csv, json, re, sys, shutil, inspect
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import yaml


YOLOV5_DIR   = Path(r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\yolov5")
WEIGHTS      = Path(r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\runs\train\exp110\weights\best.pt")
TRAIN_IMAGES = Path(r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\dataset\dataset_yolo_tiled\images\train")
OUTDIR       = Path(r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\artifacts_student\yolo_train")


IMGSZ   = 1536
CONF    = 0.10
IOU     = 0.50
DEVICE  = "0"   
MAX_DET = 300
HALF    = True
AUGMENT = False
EXIST_OK = True
CLEAN_RUN = True  


NAMES_YAML: Optional[Path] = Path(r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\dataset\dataset_yolo_tiled\data.yaml")


PROGRESS_EVERY = 500
STRICT_REQUIRE_IMAGE_SIZES = False  


def log(msg: str): print(msg, flush=True)

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def gather_images(source_dir: Optional[Path]) -> Dict[str, Tuple[int,int,str]]:
    if not source_dir or not source_dir.exists(): return {}
    exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp",".JPG",".PNG",".TIF",".WEBP"}
    size_map: Dict[str, Tuple[int,int,str]] = {}
    for idx, p in enumerate(source_dir.rglob("*")):
        if p.suffix in exts:
            try:
                with Image.open(p) as im:
                    size_map[p.stem] = (im.width, im.height, str(p.relative_to(source_dir)))
            except Exception as e:
                log(f"[WARN] Failed to open {p}: {e}")
        if (idx+1) % (PROGRESS_EVERY*10) == 0:
            log(f"[INFO] Scanned {idx+1} image paths…")
    return size_map

def load_names(names_yaml: Optional[Path], max_class_id_seen: int) -> List[Dict]:
    if not names_yaml or not names_yaml.exists():
        return [{"id": i, "name": str(i)} for i in range(max_class_id_seen+1)]
    try:
        with open(names_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        names = data.get("names")
        if isinstance(names, dict):
            cats = [{"id": int(k), "name": str(v)} for k,v in names.items()]
            max_id = max((c["id"] for c in cats), default=-1)
            if max_class_id_seen > max_id:
                present = {c["id"] for c in cats}
                for i in range(max_id+1, max_class_id_seen+1):
                    if i not in present: cats.append({"id": i, "name": str(i)})
            return sorted(cats, key=lambda x: x["id"])
        elif isinstance(names, list):
            cats = [{"id": i, "name": n} for i,n in enumerate(names)]
            if max_class_id_seen >= len(cats):
                for i in range(len(cats), max_class_id_seen+1):
                    cats.append({"id": i, "name": str(i)})
            return cats
    except Exception as e:
        log(f"[WARN] Failed to read names from {names_yaml}: {e}")
    return [{"id": i, "name": str(i)} for i in range(max_class_id_seen+1)]

def yolo_to_coco_xywh(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int):
    x = (xc - w/2.0) * img_w
    y = (yc - h/2.0) * img_h
    return x, y, w*img_w, h*img_h


def run_yolov5_detect_inprocess() -> Path:
    if not YOLOV5_DIR.exists():   raise SystemExit(f"YOLOV5_DIR not found: {YOLOV5_DIR}")
    if not WEIGHTS.exists():      raise SystemExit(f"WEIGHTS not found: {WEIGHTS}")
    if not TRAIN_IMAGES.exists(): raise SystemExit(f"TRAIN_IMAGES not found: {TRAIN_IMAGES}")

    project = OUTDIR.resolve().parent
    name = OUTDIR.name
    if CLEAN_RUN and OUTDIR.exists():
        log(f"[CLEAN] Removing existing {OUTDIR}")
        shutil.rmtree(OUTDIR, ignore_errors=True)
    ensure_dir(project)

    sys.path.insert(0, str(YOLOV5_DIR.resolve()))
    try:
        from detect import run as yolo_detect_run  
    except Exception as e:
        raise SystemExit(f"Could not import detect.py from {YOLOV5_DIR}: {e}")

    # Build kwargs and filter to only what this version supports
    kwargs = dict(
        weights=str(WEIGHTS),
        source=str(TRAIN_IMAGES),
        imgsz=(IMGSZ, IMGSZ),               
        conf_thres=CONF,
        iou_thres=IOU,
        device=DEVICE,
        max_det=MAX_DET,
        save_txt=True,
        save_conf=True,
        save_crop=True,
        project=str(project),
        name=name,
        exist_ok=EXIST_OK or CLEAN_RUN,
        half=HALF,
        augment=AUGMENT,
        line_thickness=2,
        
    )
    sig = inspect.signature(yolo_detect_run).parameters
    filtered = {k: v for k, v in kwargs.items() if k in sig}

    log("[STEP 1] Running YOLOv5 detect in-process…")
    yolo_detect_run(**filtered)

    run_dir = project / name
    labels_dir = run_dir / "labels"
    if not labels_dir.exists():
        raise SystemExit(f"detect.py finished but no labels found at {labels_dir}")
    n_txt = len(list(labels_dir.glob("*.txt")))
    log(f"[OK] YOLOv5 finished. labels/: {n_txt} files → {labels_dir}")
    return run_dir


def export_predictions(run_dir: Path):
    labels_dir = run_dir / "labels"
    out_csv       = run_dir / "predictions.csv"
    out_json_abs  = run_dir / "predictions_coco.json"
    out_json_norm = run_dir / "predictions_coco_norm.json"

    size_map = gather_images(TRAIN_IMAGES)
    if size_map:
        log(f"[STEP 2] Export: Found {len(size_map)} images under {TRAIN_IMAGES}")
    else:
        log("[STEP 2] Export: No image sizes available → absolute bboxes skipped; normalized will be written.")

    
    image_id_map: Dict[str,int] = {}
    images_abs:  List[Dict] = []
    images_norm: List[Dict] = []

    for stem, (w,h,relname) in sorted(size_map.items(), key=lambda x: x[0]):
        image_id_map[stem] = len(image_id_map) + 1
        images_abs.append({"id": image_id_map[stem], "file_name": relname, "width": w, "height": h})
        images_norm.append({"id": image_id_map[stem], "file_name": relname, "width": w, "height": h})

    txt_files = sorted(labels_dir.glob("*.txt"))
    if not txt_files:
        log(f"[WARN] No .txt files found in {labels_dir}")

    for stem in sorted({p.stem for p in txt_files}):
        if stem not in image_id_map:
            image_id_map[stem] = len(image_id_map) + 1
            images_norm.append({"id": image_id_map[stem], "file_name": stem, "width": None, "height": None})

    annotations_abs:  List[Dict] = []
    annotations_norm: List[Dict] = []
    records_for_csv:  List[Dict] = []
    ann_id_abs = 1; ann_id_norm = 1; max_cls = -1

    for idx, txt in enumerate(txt_files):
        stem = txt.stem
        known = stem in size_map
        img_w = size_map[stem][0] if known else None
        img_h = size_map[stem][1] if known else None
        image_id = image_id_map[stem]

        try:
            with open(txt, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            log(f"[WARN] Failed to read {txt}: {e}")
            continue

        for line in lines:
            line = line.strip()
            if not line: continue
            parts = re.split(r"\s+", line)
            if len(parts) < 5: continue
            try:
                cls = int(float(parts[0]))
                xc, yc, w, h = map(float, parts[1:5])
                conf = float(parts[5]) if len(parts) >= 6 else None
            except Exception:
                continue

            max_cls = max(max_cls, cls)

            
            annotations_norm.append({
                "id": ann_id_norm,
                "image_id": image_id,
                "category_id": cls,
                "bbox_norm": [xc, yc, w, h],
                **({"score": conf} if conf is not None else {}),
            }); ann_id_norm += 1

            
            if known and img_w is not None and img_h is not None:
                x, y, W, H = yolo_to_coco_xywh(xc, yc, w, h, img_w, img_h)
                annotations_abs.append({
                    "id": ann_id_abs,
                    "image_id": image_id,
                    "category_id": cls,
                    "bbox": [x, y, W, H],
                    "area": W*H,
                    "iscrowd": 0,
                    **({"score": conf} if conf is not None else {}),
                }); ann_id_abs += 1

            row = {
                "image_id": image_id, "image_stem": stem, "cls": cls,
                "conf": conf if conf is not None else "",
                "xc_norm": xc, "yc_norm": yc, "w_norm": w, "h_norm": h,
                "img_w": img_w if known else "", "img_h": img_h if known else "",
                "x":"", "y":"", "w":"", "h":""
            }
            if known and img_w is not None and img_h is not None:
                x, y, W, H = yolo_to_coco_xywh(xc, yc, w, h, img_w, img_h)
                row.update({"x": x, "y": y, "w": W, "h": H})
            records_for_csv.append(row)

        if (idx+1) % PROGRESS_EVERY == 0:
            log(f"[INFO] Parsed {idx+1}/{len(txt_files)} label files…")

    if STRICT_REQUIRE_IMAGE_SIZES and (not size_map):
        raise SystemExit("STRICT mode: missing image sizes.")

    if max_cls < 0: max_cls = 0
    categories = load_names(NAMES_YAML, max_cls)

    
    fieldnames = ["image_id","image_stem","cls","conf","xc_norm","yc_norm","w_norm","h_norm","img_w","img_h","x","y","w","h"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(records_for_csv)
    log(f"[OK] Wrote CSV: {out_csv} (rows={len(records_for_csv)})")

    
    out_abs_written = False
    
    if images_abs and annotations_abs:
        with open(out_json_abs, "w", encoding="utf-8") as f:
            json.dump({"images": images_abs, "annotations": annotations_abs, "categories": categories}, f)
        log(f"[OK] Wrote COCO JSON (absolute): {out_json_abs} (images={len(images_abs)}, anns={len(annotations_abs)})")
    else:
        log("[NOTE] Skipped COCO absolute JSON (no image sizes available or no absolute annotations)")


    
    with open(out_json_norm, "w", encoding="utf-8") as f:
        json.dump({"images": images_norm, "annotations": annotations_norm, "categories": categories}, f)
    log(f"[OK] Wrote COCO-like JSON (normalized): {out_json_norm} (images={len(images_norm)}, anns={len(annotations_norm)})")

def main():
    log("========== YOLOv5 → Export pipeline ==========")
    run_dir = run_yolov5_detect_inprocess()
    export_predictions(run_dir)
    log("[DONE] All finished.")

if __name__ == "__main__":
    main()
