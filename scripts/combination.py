#!/usr/bin/env python3

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import csv, json
import torch, yaml
from PIL import Image, ImageDraw


RSCLIP_LABELS_DIR = Path(r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\artifacts_student\rsclip_verified_train_fast\labels_verified")
TRAIN_IMAGES       = Path(r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\dataset\dataset_yolo_tiled\images")
NAMES_YAML         = Path(r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\dataset\dataset_yolo_tiled\data.yaml")
OUTDIR             = Path(r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\artifacts_student\rsclip_owlv2_refined")

OWL_MODEL_ID   = "google/owlv2-base-patch16-ensemble"
OWL_SCORE_THR  = 0.20   
IOU_MATCH_THR  = 0.50   
REFINEMENT_MODE = "snap_to_owl" 
FUSE_ALPHA     = 0.5
FAILSAFE_KEEP  = True

MAKE_OVERLAYS  = True
OVERLAY_EVERY  = 200
LINE_W         = 3
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

PROMPT_TEMPLATES = [
    "a satellite image of a {}",
    "an aerial photo of a {}",
    "an overhead view of a {}",
    "a top-down remote sensing image of a {}",
]


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_names(p: Path) -> Dict[int,str]:
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    names = data.get("names")
    if isinstance(names, dict):  return {int(k): str(v) for k,v in names.items()}
    if isinstance(names, list):  return {i: str(n) for i,n in enumerate(names)}
    raise ValueError("Bad data.yaml: names must be list or dict")

def index_images(root: Path) -> Dict[str,Path]:
    exts = {".jpg",".jpeg",".png",".tif",".tiff",".bmp",".webp",".JPG",".PNG",".TIF",".WEBP"}
    return {p.stem: p for p in root.rglob("*") if p.suffix in exts}

def read_yolo_5or6(txt: Path):
    rows=[]
    if not txt.exists(): return rows
    for ln in txt.read_text(encoding="utf-8").splitlines():
        parts = ln.strip().split()
        if len(parts) < 5: continue
        cls=int(float(parts[0])); xc,yc,w,h = map(float, parts[1:5])
        rows.append((cls,xc,yc,w,h))
    return rows

def write_yolo_5(path: Path, rows: List[Tuple[int,float,float,float,float]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path,"w",encoding="utf-8") as f:
        for cls,xc,yc,w,h in rows:
            f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

def yolo_to_xyxy(xc,yc,w,h,W,H):
    return ( (xc-w/2)*W, (yc-h/2)*H, (xc+w/2)*W, (yc+h/2)*H )

def xyxy_to_yolo(x1,y1,x2,y2,W,H):
    w=(x2-x1)/W; h=(y2-y1)/H; xc=(x1+x2)/(2*W); yc=(y1+y2)/(2*H)
    return xc,yc,w,h

def iou_xyxy(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1=max(ax1,bx1); iy1=max(ay1,by1); ix2=min(ax2,bx2); iy2=min(ay2,by2)
    iw=max(0.0,ix2-ix1); ih=max(0.0,iy2-iy1); inter=iw*ih
    aA=max(0.0,ax2-ax1)*max(0.0,ay2-ay1); bA=max(0.0,bx2-bx1)*max(0.0,by2-by1)
    denom=aA+bA-inter
    return inter/denom if denom>0 else 0.0

def draw_boxes(img: Image.Image, boxes, color):
    im=img.copy(); d=ImageDraw.Draw(im)
    for x1,y1,x2,y2 in boxes:
        d.rectangle((x1,y1,x2,y2), outline=color, width=LINE_W)
    return im


def main():
    
    for p in [RSCLIP_LABELS_DIR, TRAIN_IMAGES, NAMES_YAML]:
        if not p.exists(): raise FileNotFoundError(p)
    ensure_dir(OUTDIR)
    out_labels = ensure_dir(OUTDIR / "labels")
    out_audit  = OUTDIR / "audit.csv"
    out_stats  = OUTDIR / "stats.json"
    out_vis    = ensure_dir(OUTDIR / "overlays") if MAKE_OVERLAYS else None

    id2name = load_names(NAMES_YAML)
    img_by_stem = index_images(TRAIN_IMAGES)

    
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    if "cuda" in DEVICE and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif "cuda" in DEVICE:
        dtype = torch.float16
    else:
        dtype = torch.float32

    print(f"Loading {OWL_MODEL_ID} on {DEVICE} (dtype={dtype})")
    processor = Owlv2Processor.from_pretrained(OWL_MODEL_ID)
    model = Owlv2ForObjectDetection.from_pretrained(OWL_MODEL_ID, torch_dtype=dtype).to(DEVICE).eval()

    kept = refined = dropped = 0
    audit_rows = []

    txt_files = sorted(RSCLIP_LABELS_DIR.glob("*.txt"))
    for i, tpath in enumerate(txt_files):
        stem = tpath.stem
        imgp = img_by_stem.get(stem)
        if not imgp: continue
        im = Image.open(imgp).convert("RGB")
        W,H = im.size
        labels = read_yolo_5or6(tpath)
        if not labels:
            write_yolo_5(out_labels / f"{stem}.txt", [])
            continue

        
        cls_present = sorted({cls for cls, *_ in labels})
        prompts=[]
        for cid in cls_present:
            nm = id2name.get(cid, str(cid)).lower()
            for t in PROMPT_TEMPLATES:
                prompts.append(t.format(nm))
        
        seen=set(); uniq=[]
        for p in prompts:
            if p not in seen:
                uniq.append(p); seen.add(p)
        text_labels=[uniq[:80]]

        with torch.no_grad():
            inputs = processor(text=text_labels, images=im, return_tensors="pt").to(DEVICE)
            outputs = model(**inputs)
            
            target_sizes = torch.tensor([(H, W)], dtype=torch.int64, device=DEVICE)
            results = processor.post_process_grounded_object_detection(
                outputs, target_sizes=target_sizes, threshold=OWL_SCORE_THR, text_labels=text_labels
            )[0]

        
        boxes_t  = results["boxes"].detach().to(dtype=torch.float32, device="cpu")
        scores_t = results["scores"].detach().to(dtype=torch.float32, device="cpu")

        owl_boxes  = [tuple(map(float, b)) for b in boxes_t.numpy()]
        owl_scores = [float(s) for s in scores_t.numpy()]
        owl_texts  = [str(t) for t in results["text_labels"]]
        

        
        by_text = {}
        for b,s,t in zip(owl_boxes, owl_scores, owl_texts):
            by_text.setdefault(t, []).append((b,s))

        new_rows=[]
        for (cls,xc,yc,w,h) in labels:
            nm = id2name.get(cls, str(cls)).lower()
            
            cands=[]
            for t in PROMPT_TEMPLATES:
                cands += by_text.get(t.format(nm), [])
            tx1,ty1,tx2,ty2 = yolo_to_xyxy(xc,yc,w,h,W,H)

            
            best_iou=0.0; best=None
            for (bxyxy,score) in cands:
                if score < OWL_SCORE_THR:
                    continue
                iou = iou_xyxy((tx1,ty1,tx2,ty2), bxyxy)
                if iou > best_iou:
                    best_iou = iou; best = (bxyxy, score)

            if best and best_iou >= IOU_MATCH_THR:
                (ox1,oy1,ox2,oy2), sc = best
                if REFINEMENT_MODE == "snap_to_owl":
                    x1,y1,x2,y2 = ox1,oy1,ox2,oy2; src="owl"
                elif REFINEMENT_MODE == "fuse":
                    x1,y1,x2,y2 = 0.5*(tx1+ox1), 0.5*(ty1+oy1), 0.5*(tx2+ox2), 0.5*(ty2+oy2); src="fused"
                else:
                    x1,y1,x2,y2 = tx1,ty1,tx2,ty2; src="teacher"
                refined += 1; kept += 1
                xc2,yc2,w2,h2 = xyxy_to_yolo(x1,y1,x2,y2,W,H)
                new_rows.append((cls,xc2,yc2,w2,h2))
                audit_rows.append({"stem":stem,"cls":cls,"label":nm,"decision":"keep_refined","iou":f"{best_iou:.3f}","src":src,"owl_score":f"{sc:.3f}"})
            else:
                if FAILSAFE_KEEP:
                    kept += 1
                    new_rows.append((cls,xc,yc,w,h))
                    audit_rows.append({"stem":stem,"cls":cls,"label":nm,"decision":"keep_failsafe","iou":f"{best_iou:.3f}","src":"teacher","owl_score":""})
                else:
                    dropped += 1
                    audit_rows.append({"stem":stem,"cls":cls,"label":nm,"decision":"drop","iou":f"{best_iou:.3f}","src":"","owl_score":""})

        write_yolo_5(out_labels / f"{stem}.txt", new_rows)

        
        if MAKE_OVERLAYS and (OVERLAY_EVERY==0 or i % OVERLAY_EVERY == 0):
            t_boxes = [yolo_to_xyxy(xc,yc,w,h,W,H) for (_,xc,yc,w,h) in labels]
            r_boxes = [yolo_to_xyxy(xc,yc,w,h,W,H) for (_,xc,yc,w,h) in new_rows]
            vis = draw_boxes(im, t_boxes, "red")
            vis = draw_boxes(vis, r_boxes, "lime")
            vis.save((OUTDIR / "overlays" / f"{stem}.jpg"))

        if (i+1) % 100 == 0:
            print(f"[{i+1}/{len(txt_files)}] kept={kept} refined={refined} dropped={dropped}")

    
    with open(out_audit,"w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=["stem","cls","label","decision","iou","src","owl_score"])
        writer.writeheader(); writer.writerows(audit_rows)
    stats = {"kept": kept, "refined": refined, "dropped": dropped, "images": len(txt_files),
             "OWL_SCORE_THR": OWL_SCORE_THR, "IOU_MATCH_THR": IOU_MATCH_THR, "FAILSAFE_KEEP": FAILSAFE_KEEP}
    with open(out_stats,"w",encoding="utf-8") as f:
        json.dump(stats,f,indent=2)
    print("\n[DONE] RS-CLIP → OWLv2 refinement complete.")
    print("Labels  →", out_labels)
    print("Audit   →", out_audit)
    print("Stats   →", out_stats)
    if MAKE_OVERLAYS: print("Overlays→", OUTDIR/"overlays")

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True
    main()
