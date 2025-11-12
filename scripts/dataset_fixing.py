#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import os, shutil, yaml, csv
from PIL import Image


RAW_ROOT       = Path(r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\dataset\dataset_yolo_tiled")
RAW_IMG_TRAIN  = RAW_ROOT / "images" / "train"
RAW_IMG_VAL    = RAW_ROOT / "images" / "val"
RAW_LBL_TRAIN  = RAW_ROOT / "labels" / "train"   
RAW_LBL_VAL    = RAW_ROOT / "labels" / "val"     
RAW_DATA_YAML  = RAW_ROOT / "data.yaml"

VARIANTS = [
    ("teacher_only",         Path(r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\artifacts_student\yolo_train\labels")),
    ("fp_filter_only",       Path(r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\artifacts_student\rsclip_verified_train_fast\labels_verified")),
    ("refine_only",          Path(r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\artifacts_student\owlv2_visualgrounding_artifacts\owlv2_grounded_train\labels_grounded")),
    ("fp_filter_plus_refine",Path(r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\artifacts_student\rsclip_owlv2_refined\labels")),
]



OUT_BASE = Path(r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\artifacts_student\student_datasets_all")


USE_HARDLINKS        = True
TRIM_TO_5COL         = True
BACKFILL_MISSING     = True      
DEDUP_IOU_THR        = 0.50      
ALLOWED_IMG_EXTS     = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp",".JPG",".PNG",".TIF",".WEBP"}


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def copy_or_hardlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        if USE_HARDLINKS:
            os.link(src, dst)
        else:
            shutil.copy2(src, dst)
    except Exception:
        shutil.copy2(src, dst)

def load_names(yaml_path: Path) -> Dict[int,str]:
    y = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = y.get("names")
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    raise ValueError("data.yaml 'names' must be list or dict")

def list_images(root: Path) -> Dict[str, Path]:
    idx = {}
    for p in root.rglob("*"):
        if p.suffix in ALLOWED_IMG_EXTS:
            idx[p.stem] = p
    return idx

def read_yolo_5or6(txt: Path) -> List[Tuple[int,float,float,float,float]]:
    rows = []
    if not txt.exists():
        return rows
    for ln in txt.read_text(encoding="utf-8").splitlines():
        parts = ln.strip().split()
        if len(parts) >= 5:
            try:
                cls = int(float(parts[0])); xc, yc, w, h = map(float, parts[1:5])
                rows.append((cls, xc, yc, w, h))
            except Exception:
                continue
    return rows

def write_yolo_5(path: Path, rows: List[Tuple[int,float,float,float,float]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for cls, xc, yc, w, h in rows:
            f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

def yolo_to_xyxy(xc,yc,w,h,W,H):
    return ( (xc - w/2) * W, (yc - h/2) * H, (xc + w/2) * W, (yc + h/2) * H )

def iou_xyxy(a,b) -> float:
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1 = max(ax1,bx1); iy1 = max(ay1,by1); ix2 = min(ax2,bx2); iy2 = min(ay2,by2)
    iw = max(0.0, ix2-ix1); ih = max(0.0, iy2-iy1)
    inter = iw*ih
    aA = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
    bA = max(0.0, bx2-bx1) * max(0.0, by2-by1)
    denom = aA + bA - inter
    return inter/denom if denom > 0 else 0.0

def count_classes(labels_dir: Path) -> Dict[int,int]:
    from collections import Counter
    c = Counter()
    if not labels_dir.exists():
        return c
    for p in labels_dir.glob("*.txt"):
        for cls, *_ in read_yolo_5or6(p):
            c[cls] += 1
    return c

def compute_val_present(val_dir: Path) -> set[int]:
    present = set()
    if not val_dir.exists():
        return present
    for p in val_dir.glob("*.txt"):
        for cls, *_ in read_yolo_5or6(p):
            present.add(cls)
    return present

def backfill_missing_classes(
    out_train_img: Path,
    out_train_lbl: Path,
    variant_present: set[int],
    val_present: set[int],
    raw_gt_train_dir: Path,
    raw_train_img_index: Dict[str, Path],
    iou_thr: float = 0.5
):
    missing = sorted(val_present - variant_present)
    if not missing:
        return {"added_boxes": 0, "images_touched": 0, "missing_classes": []}

    added = 0
    touched = 0

    
    for gt_path in raw_gt_train_dir.glob("*.txt"):
        stem = gt_path.stem
        cand_rows = [r for r in read_yolo_5or6(gt_path) if r[0] in missing]
        if not cand_rows:
            continue

        out_lbl_path = out_train_lbl / f"{stem}.txt"
        exist_rows = read_yolo_5or6(out_lbl_path)
        
        img_path = raw_train_img_index.get(stem)
        if img_path is None:
            continue
        W, H = Image.open(img_path).size

        
        exist_by_cls = {}
        for cls, xc, yc, w, h in exist_rows:
            exist_by_cls.setdefault(cls, []).append(yolo_to_xyxy(xc,yc,w,h,W,H))

        new_rows = list(exist_rows)
        appended_any = False
        for cls, xc, yc, w, h in cand_rows:
            box_new = yolo_to_xyxy(xc,yc,w,h,W,H)
            dup = False
            for box_old in exist_by_cls.get(cls, []):
                if iou_xyxy(box_new, box_old) >= iou_thr:
                    dup = True
                    break
            if not dup:
                new_rows.append((cls, xc, yc, w, h))
                exist_by_cls.setdefault(cls, []).append(box_new)
                added += 1
                appended_any = True

        if appended_any:
            write_yolo_5(out_lbl_path, new_rows)
            touched += 1

    return {"added_boxes": added, "images_touched": touched, "missing_classes": missing}


def main():
    
    for p in [RAW_IMG_TRAIN, RAW_IMG_VAL, RAW_LBL_TRAIN, RAW_LBL_VAL, RAW_DATA_YAML]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    id2name = load_names(RAW_DATA_YAML)
    raw_train_imgs = list_images(RAW_IMG_TRAIN)

    
    val_present = compute_val_present(RAW_LBL_VAL)

    
    for name, var_lbl_dir in VARIANTS:
        if not var_lbl_dir.exists():
            print(f"[{name}] SKIP: labels dir does not exist: {var_lbl_dir}")
            continue

        out_root      = OUT_BASE / name
        out_train_img = ensure_dir(out_root / "images" / "train")
        out_train_lbl = ensure_dir(out_root / "labels" / "train")
        data_yaml_out = out_root / f"data_{name}.yaml"

        print(f"\n=== Building variant: {name} ===")
        print(f"labels_in = {var_lbl_dir}")
        print(f"out_root  = {out_root}")

        
        img_idx = list_images(RAW_IMG_TRAIN)
        print(f"[{name}] Mirroring train images … ({len(img_idx)} files)")
        for stem, src in img_idx.items():
            copy_or_hardlink(src, out_train_img / src.name)

        
        print(f"[{name}] Copying/normalizing labels …")
        n_files = 0; n_boxes = 0
        for stem, _ in img_idx.items():
            src_lbl = var_lbl_dir / f"{stem}.txt"
            dst_lbl = out_train_lbl / f"{stem}.txt"
            if src_lbl.exists():
                rows = read_yolo_5or6(src_lbl) if TRIM_TO_5COL else read_yolo_5or6(src_lbl)
                write_yolo_5(dst_lbl, rows)
                n_files += 1; n_boxes += len(rows)
            else:
                
                write_yolo_5(dst_lbl, [])
                n_files += 1
        print(f"[{name}] Train labels written: files={n_files}, boxes={n_boxes}")

        
        var_counts = count_classes(out_train_lbl)
        var_present = {cid for cid, c in var_counts.items() if c > 0}
        missing = sorted(val_present - var_present)
        if missing:
            print(f"[{name}] Missing vs val: {len(missing)} classes")
            for cid in missing:
                print(f"  - {cid:>3} : {id2name.get(cid,'UNKNOWN')}")
        else:
            print(f"[{name}] No missing classes vs validation.")

        
        backfill_stats = {"added_boxes": 0, "images_touched": 0, "missing_classes": []}
        if BACKFILL_MISSING and missing:
            print(f"[{name}] Backfilling missing classes from RAW train GT (IoU dedup ≥ {DEDUP_IOU_THR}) …")
            backfill_stats = backfill_missing_classes(
                out_train_img=out_train_img,
                out_train_lbl=out_train_lbl,
                variant_present=var_present,
                val_present=val_present,
                raw_gt_train_dir=RAW_LBL_TRAIN,
                raw_train_img_index=raw_train_imgs,
                iou_thr=DEDUP_IOU_THR
            )
            print(f"[{name}] Backfill done: +{backfill_stats['added_boxes']} boxes in {backfill_stats['images_touched']} images")

        
        nc = len(id2name)
        names_list = [id2name[i] for i in range(nc)]
        data = {
            "path": out_root.as_posix(),
            "train": "images/train",
            "val": RAW_IMG_VAL.as_posix(),   
            "nc": nc,
            "names": names_list,
        }
        data_yaml_out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        print(f"[{name}] Wrote: {data_yaml_out}")

        
        summary_csv = out_root / "summary.csv"
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["variant","train_files","train_boxes","missing_classes_count","backfill_added_boxes","backfill_images_touched"])
            w.writerow([name, n_files, n_boxes, len(missing), backfill_stats["added_boxes"], backfill_stats["images_touched"]])
        print(f"[{name}] Wrote: {summary_csv}")

        
        runs_dir = out_root / "runs"
        print("\nTrain this variant with YOLOv5, e.g.:")
        print(rf"""conda activate yolo_env
python "{(RAW_ROOT.parent / 'yolo_pipeline' / 'yolov5' / 'train.py')}" --img 1024 --batch 8 --epochs 50 ^
  --data "{data_yaml_out}" ^
  --weights yolov5m.pt --project "{runs_dir.as_posix()}" --name "{name}" --exist-ok""")

    print("\n=== ALL VARIANTS DONE ===")
    print("Outputs in:", OUT_BASE)

if __name__ == "__main__":
    main()
