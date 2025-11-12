#!/usr/bin/env python3
from pathlib import Path
import json, csv, hashlib, shutil
from collections import defaultdict, Counter
from PIL import Image
import numpy as np


DATASET_IN       = Path("C:/Users/elpob/Documents/Projects/new_pipeline/yolo_pipeline/dataset")
GEOJSON_PATH     = Path("C:/Users/elpob/Documents/Projects/new_pipeline/yolo_pipeline/xView_train.geojson")
CLASS_LIST_JSON  = Path("C:/Users/elpob/Documents/Projects/new_pipeline/yolo_pipeline/scripts/class_list.json")

RAW_TRAIN_DIR    = DATASET_IN / "train_images"
RAW_VAL_DIR      = DATASET_IN / "val_images"

OUT_ROOT         = DATASET_IN / "dataset_yolo"
OUT_IMG          = OUT_ROOT / "images"
OUT_LBL          = OUT_ROOT / "labels"
OUT_DATA_YAML    = OUT_ROOT / "data.yaml"
WRITE_REPORT_CSV = OUT_ROOT / "preprocess_report.csv"
CLASS_DIST_CSV   = OUT_ROOT / "class_distribution.csv"


SPLIT_RATIOS     = {"train": 0.75, "val": 0.15, "test": 0.10}
SPLIT_SEED       = 42            

# Image conversion
CONVERT_TO_JPG   = True
JPG_QUALITY      = 92            
CLEAN_OUT_ROOT   = True          


def load_class_mapping(class_list_path: Path):
    data = json.loads(class_list_path.read_text(encoding="utf-8"))
    if "classes" not in data:
        raise ValueError("class_list.json must have a 'classes' list")
    yolo_to_name, xview_to_yolo = {}, {}
    for c in data["classes"]:
        yi, xid, name = int(c["yolo_idx"]), int(c["xview_id"]), c["name"]
        yolo_to_name[yi] = name
        xview_to_yolo[xid] = yi
    return yolo_to_name, xview_to_yolo

def tif_to_jpg(in_path: Path, out_path: Path, quality=92):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(in_path) as im:
        
        if im.mode in ("RGB", "L"):
            im8 = im
        elif im.mode.startswith("I;16") or im.mode == "I" or "16" in im.mode:
            arr = np.array(im, dtype=np.float32)
            lo, hi = float(arr.min()), float(arr.max())
            scale = 255.0 / max(1e-6, hi - lo)
            arr8 = np.clip((arr - lo) * scale, 0, 255).astype(np.uint8)
            im8 = Image.fromarray(arr8, mode="L")
        else:
            im8 = im.convert("RGB")
        if im8.mode != "RGB":
            im8 = im8.convert("RGB")
        im8.save(out_path, format="JPEG", quality=quality, optimize=True)
        return im8.size  

def clip01(v):  
    return min(max(v, 0.0), 1.0)

def yolo_bbox_from_bounds(bounds_imcoords, img_w, img_h):
    
    if isinstance(bounds_imcoords, str):
        x1, y1, x2, y2 = [float(x) for x in bounds_imcoords.split(",")]
    else:
        x1, y1, x2, y2 = [float(v) for v in bounds_imcoords]
    
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(img_w), x2), min(float(img_h), y2)
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return None
    x_c = (x1 + x2) / 2.0 / img_w
    y_c = (y1 + y2) / 2.0 / img_h
    w_n = w / img_w
    h_n = h / img_h
    return (clip01(x_c), clip01(y_c), clip01(w_n), clip01(h_n))

def collect_geojson_annotations(geojson_path: Path):
    gj = json.loads(geojson_path.read_text(encoding="utf-8"))
    by_image = defaultdict(list)
    for feat in gj["features"]:
        props = feat.get("properties", {})
        raw_id = props.get("image_id") or props.get("image") or props.get("uid")
        if not raw_id:
            continue
        stem = Path(Path(str(raw_id)).name).stem 
        type_id = props.get("type_id")
        bounds  = props.get("bounds_imcoords")
        if type_id is None or bounds in (None, "", "EMPTY"):
            continue
        by_image[stem].append((int(type_id), bounds))
    return by_image

def ensure_dirs():
    if CLEAN_OUT_ROOT and OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    for split in ("train", "val", "test"):
        (OUT_IMG / split).mkdir(parents=True, exist_ok=True)
        (OUT_LBL / split).mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

def resolve_input_images():
    
    candidates = {}
    def add_dir(d):
        if d and d.exists():
            for p in d.glob("*.tif"):
                candidates[p.name] = p
    add_dir(RAW_TRAIN_DIR)
    add_dir(RAW_VAL_DIR)
    return candidates  

def hash_bucket(name: str, seed: int = 42) -> float:
    h = hashlib.md5((str(seed) + name).encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF  

def split_by_hash(filenames):
    train_set, val_set, test_set = set(), set(), set()
    for name in filenames:
        r = hash_bucket(name, SPLIT_SEED)
        if r < SPLIT_RATIOS["train"]:
            train_set.add(name)
        elif r < SPLIT_RATIOS["train"] + SPLIT_RATIOS["val"]:
            val_set.add(name)
        else:
            test_set.add(name)
    return train_set, val_set, test_set

def write_label(lines, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

def generate_data_yaml(yolo_to_name: dict):
    names = [yolo_to_name[i] for i in sorted(yolo_to_name.keys())]
    yaml_path_str = str(OUT_ROOT).replace("\\", "/")
    yaml = [
        f"path: {yaml_path_str}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(names)}",
        "names:"
    ] + [f"  - {n}" for n in names]
    OUT_DATA_YAML.write_text("\n".join(yaml) + "\n", encoding="utf-8")


def process_split(split_name, split_set, tif_index, ann_by_image, xview_to_yolo,
                  report_rows, class_counter):
    num_imgs, num_objs, num_empty = 0, 0, 0
    for fname in sorted(split_set):
        tif_path = tif_index.get(fname)
        if not tif_path:
            continue
        stem = Path(fname).stem
        out_img = OUT_IMG / split_name / f"{stem}.jpg"
        
        if CONVERT_TO_JPG:
            img_w, img_h = tif_to_jpg(tif_path, out_img, quality=JPG_QUALITY)
        else:
            
            out_img = OUT_IMG / split_name / f"{stem}.tif"
            shutil.copy2(tif_path, out_img)
            with Image.open(tif_path) as im_check:
                img_w, img_h = im_check.size

        
        out_lbl = OUT_LBL / split_name / f"{stem}.txt"
        anns = ann_by_image.get(stem, [])
        lines = []
        for type_id, bounds in anns:
            if type_id not in xview_to_yolo:
                
                continue
            yolo_cls = xview_to_yolo[type_id]
            yb = yolo_bbox_from_bounds(bounds, img_w, img_h)
            if yb is None:
                continue
            x_c, y_c, w_n, h_n = yb
            lines.append(f"{yolo_cls} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")
            class_counter[yolo_cls] += 1
        
        write_label(lines, out_lbl)
        num_imgs += 1
        num_objs += len(lines)
        if len(lines) == 0:
            num_empty += 1
        report_rows.append({
            "split": split_name,
            "image": fname,
            "num_objects": len(lines),
            "output_image": str(out_img.relative_to(OUT_ROOT)),
            "output_label": str(out_lbl.relative_to(OUT_ROOT))
        })
    return num_imgs, num_objs, num_empty

def main():
    print("[INFO] Preparing output dirs…")
    ensure_dirs()

    print("[INFO] Loading class mapping…")
    yolo_to_name, xview_to_yolo = load_class_mapping(CLASS_LIST_JSON)

    print("[INFO] Parsing GeoJSON annotations…")
    ann_by_image = collect_geojson_annotations(GEOJSON_PATH)

    print("[INFO] Indexing .tif images…")
    tif_index = resolve_input_images()
    all_names = sorted(tif_index.keys())
    if not all_names:
        raise RuntimeError(f"No .tif images found under {DATASET_IN}")

    
    total_ratio = sum(SPLIT_RATIOS.values())
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"SPLIT_RATIOS must sum to 1.0, got {total_ratio}")
    print(f"[INFO] Building splits (seed={SPLIT_SEED})…")
    train_set, val_set, test_set = split_by_hash(all_names)

    
    report_rows = []
    class_counter = Counter()
    stats = {}

    for split_name, split_set in (("train", train_set), ("val", val_set), ("test", test_set)):
        print(f"[INFO] Processing {split_name}: {len(split_set)} images")
        imgs, objs, empties = process_split(
            split_name, split_set, tif_index, ann_by_image, xview_to_yolo,
            report_rows, class_counter
        )
        stats[split_name] = {"images": imgs, "objects": objs, "empty_images": empties}
        print(f"       -> {imgs} images | {objs} objects | {empties} empty images")

    
    with open(WRITE_REPORT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["split", "image", "num_objects", "output_image", "output_label"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    with open(CLASS_DIST_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "class_name", "count"])
        for cid in sorted(class_counter.keys()):
            w.writerow([cid, yolo_to_name.get(cid, "UNKNOWN"), class_counter[cid]])

    generate_data_yaml(yolo_to_name)

    
    tot_images = sum(v["images"] for v in stats.values())
    tot_objs   = sum(v["objects"] for v in stats.values())
    tot_empty  = sum(v["empty_images"] for v in stats.values())
    print("\n[SUMMARY]")
    for s in ("train", "val", "test"):
        v = stats[s]
        print(f"  {s:5s} -> images: {v['images']:5d} | objects: {v['objects']:7d} | empty: {v['empty_images']:5d}")
    print(f"  TOTAL -> images: {tot_images:5d} | objects: {tot_objs:7d} | empty: {tot_empty:5d}")
    print(f"\n[INFO] data.yaml:          {OUT_DATA_YAML}")
    print(f"[INFO] report CSV:         {WRITE_REPORT_CSV}")
    print(f"[INFO] class distribution: {CLASS_DIST_CSV}")
    print("[INFO] Preprocessing complete.")

if __name__ == "__main__":
    main()
