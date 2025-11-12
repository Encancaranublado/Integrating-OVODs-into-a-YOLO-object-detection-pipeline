#!/usr/bin/env python3
from __future__ import annotations
import os, csv, json, sys, traceback, math
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image, ImageDraw
from tqdm import tqdm
import torch
import yaml

torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True

try:
    import open_clip
except ImportError as e:
    print("FATAL: open-clip-torch is not installed.", flush=True)
    print("Install with: pip install open_clip_torch torchvision", flush=True)
    raise


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B-16"                 
MODEL_PRETRAINED = "laion2b_s34b_b88k"  

TEACHER_LABELS_DIR = Path(r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\artifacts_student\yolo_train\labels")
TRAIN_IMAGES       = Path(r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\dataset\dataset_yolo_tiled\images\train")
NAMES_YAML         = Path(r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\dataset\dataset_yolo_tiled\data.yaml")
OUTDIR             = Path(r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\artifacts_student\rsclip_verified_train_fast")

SAVE_JSON            = False   
SAVE_BOXED_EXAMPLES  = 8       
INITIAL_BATCH_SIZE   = 256     
SECOND_PASS_CONTEXT  = True
DRAW_WIDTH           = 4
WRITE_EMPTY_ON_ALL_DROPPED = True  



TEACHER_CONF_KEEP      = 0.40  
SMALL_BOX_MIN_PIX      = 6     
KEEP_MIN_PROB          = 0.12  
DROP_MAX_PROB          = 0.02  
STRONG_MARGIN          = 0.22  
CONTEXT_WEIGHT         = 0.6   


TARGET_DROPS = 50000           
USE_AMBER_BUCKET = True        
AMBER_TGT_MAX = 0.04           
AMBER_MARGIN_MIN = 0.18        
MAX_DROP_FRACTION_PER_IMAGE = 0.50  

PROMPT_TEMPLATES = [
    "a satellite image of a {}",
    "an aerial photo of a {}",
    "an overhead view of a {}",
    "a top-down remote sensing image of a {}",
    "a high-altitude view of a {}",
]


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def load_names(names_yaml: Path) -> Dict[int,str]:
    with open(names_yaml,"r",encoding="utf-8") as f:
        data=yaml.safe_load(f)
    names=data.get("names")
    if isinstance(names,dict):  return {int(k):str(v) for k,v in names.items()}
    if isinstance(names,list):  return {i:str(n) for i,n in enumerate(names)}
    raise ValueError("Bad data.yaml: expected list or dict for 'names'")

def read_teacher_labels(txt_path: Path):
    rows=[]
    with open(txt_path,"r",encoding="utf-8") as f:
        for line in f:
            p=line.strip().split()
            if len(p)<5: continue
            cls=int(float(p[0])); xc,yc,w,h=map(float,p[1:5])
            conf=float(p[5]) if len(p)>=6 else None
            rows.append((cls,xc,yc,w,h,conf))
    return rows

def crop_from_yolo(img: Image.Image, xc,yc,w,h):
    W,H=img.size
    x1=(xc-w/2)*W; y1=(yc-h/2)*H
    x2=(xc+w/2)*W; y2=(yc+h/2)*H
    x1,y1,x2,y2=map(int,[x1,y1,x2,y2])
    x1=max(0,x1); y1=max(0,y1); x2=min(W,x2); y2=min(H,y2)
    return img.crop((x1,y1,x2,y2)), (x1,y1,x2,y2), (W,H)

def draw_box(image: Image.Image, box, color="red", width=3):
    img = image.copy()
    ImageDraw.Draw(img).rectangle(box, outline=color, width=width)
    return img

def softmax(x: torch.Tensor, dim=-1):
    x = x - x.max(dim=dim, keepdim=True).values
    return (x.exp() / x.exp().sum(dim=dim, keepdim=True))


class RSCLIP:
    def __init__(self, model_name: str, pretrained: str, device: str = "cuda"):
        print(f"[LOAD] OpenCLIP {model_name} ({pretrained}) on {device}", flush=True)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.device = device
        self.model.eval()
        # Temperature
        if hasattr(self.model, "logit_scale"):
            with torch.no_grad():
                self.logit_scale = float(self.model.logit_scale.exp().detach().cpu().item())
        else:
            self.logit_scale = 1.0

    @torch.no_grad()
    def encode_images_batched(self, pil_images: List[Image.Image], batch_size: int) -> torch.Tensor:
        if not pil_images:
            out_dim = getattr(self.model.visual, "output_dim", 512)
            return torch.empty(0, out_dim, device=self.device)
        bs = batch_size; out=[]; i=0
        while i < len(pil_images):
            j = min(i + bs, len(pil_images))
            chunk = pil_images[i:j]
            try:
                batch = torch.stack([self.preprocess(im).to(self.device, non_blocking=True) for im in chunk], dim=0)
                feats = self.model.encode_image(batch)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                out.append(feats); i=j
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and bs > 8 and self.device.startswith("cuda"):
                    torch.cuda.empty_cache(); bs = max(8, bs // 2)
                    print(f"[OOM] Reducing batch size to {bs}", flush=True)
                else:
                    raise
        return torch.cat(out, dim=0)

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        toks = self.tokenizer(texts).to(self.device)
        feats = self.model.encode_text(toks)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats


def build_text_embeds(rsclip: RSCLIP, class_names: Dict[int,str]) -> torch.Tensor:
    prompts=[]; cls_ranges=[]
    for cid in range(len(class_names)):
        name = class_names[cid]; start=len(prompts)
        for t in PROMPT_TEMPLATES: prompts.append(t.format(name))
        end=len(prompts); cls_ranges.append((start,end))
    all_embeds = rsclip.encode_texts(prompts)  # [P,D]
    per_class=[]
    for s,e in cls_ranges:
        m = all_embeds[s:e].mean(0, keepdim=True)
        m = m / m.norm(dim=-1, keepdim=True)
        per_class.append(m)
    return torch.cat(per_class, dim=0)   # [C,D]


def first_pass_metrics(cls: int, logits_1d: torch.Tensor):
    probs = softmax(logits_1d, dim=0)
    tgt = float(probs[cls].item())
    best_val, best_idx = float(probs.max().item()), int(probs.argmax().item())
    margin = best_val - tgt if best_idx != cls else tgt - float((probs.topk(2).values[1] if probs.numel()>1 else 0.0))
    return tgt, best_val, best_idx, margin

def is_red_drop(tgt: float, margin: float, best_idx: int, cls: int) -> bool:
    
    return (best_idx != cls) and (tgt <= DROP_MAX_PROB) and (margin >= STRONG_MARGIN)

def is_amber_drop(tgt: float, margin: float, best_idx: int, cls: int) -> bool:
    
    return (best_idx != cls) and (tgt <= AMBER_TGT_MAX) and (margin >= AMBER_MARGIN_MIN)


def main():
    print("=== RS-CLIP fast filter (less conservative) — starting ===", flush=True)
    print(f"DEVICE={DEVICE} | model={MODEL_NAME}/{MODEL_PRETRAINED}", flush=True)
    print(f"labels={TEACHER_LABELS_DIR}", flush=True)
    print(f"images={TRAIN_IMAGES}", flush=True)
    print(f"names_yml={NAMES_YAML}", flush=True)
    print(f"Target drops: {TARGET_DROPS} (amber enabled: {USE_AMBER_BUCKET})", flush=True)

    
    if not TEACHER_LABELS_DIR.exists(): print("ERROR: TEACHER_LABELS_DIR does not exist.", flush=True); return
    if not TRAIN_IMAGES.exists():       print("ERROR: TRAIN_IMAGES does not exist.", flush=True); return
    if not NAMES_YAML.exists():         print("ERROR: NAMES_YAML does not exist.", flush=True); return

    ensure_dir(OUTDIR)
    out_labels = ensure_dir(OUTDIR / "labels_verified")
    out_audit  = OUTDIR / "audit.csv"
    out_boxed  = ensure_dir(OUTDIR / "boxed_examples") if SAVE_BOXED_EXAMPLES > 0 else None
    out_json   = OUTDIR / "audit.json"

    
    txt_files = sorted(TEACHER_LABELS_DIR.glob("*.txt"))
    print(f"Found {len(txt_files)} label files.", flush=True)
    if len(txt_files) == 0: print("Nothing to do. Exiting.", flush=True); return

    
    id2name = load_names(NAMES_YAML)
    print(f"Loaded {len(id2name)} classes from data.yaml", flush=True)

   
    print("Indexing training images by stem...", flush=True)
    img_index: Dict[str, Path] = {}
    for ext in ("*.jpg","*.jpeg","*.png","*.tif","*.tiff"):
        for p in TRAIN_IMAGES.rglob(ext):
            img_index[Path(p).stem] = p
    print(f"Indexed {len(img_index)} images.", flush=True)

    
    rsclip = RSCLIP(MODEL_NAME, MODEL_PRETRAINED, DEVICE)
    class_embeds = build_text_embeds(rsclip, id2name)
    print(f"Text embeddings built. logit_scale={rsclip.logit_scale:.3f}", flush=True)

    kept_boxes = 0
    dropped_boxes = 0
    rechecked  = 0

    audit_rows = []
    audit_blob = {}
    saved_boxes = 0

    with torch.inference_mode():
        for txt_path in tqdm(txt_files, desc="RS-CLIP filtering", unit="image"):
            stem = txt_path.stem
            img_path = img_index.get(stem)
            if not img_path: continue
            image = Image.open(img_path).convert("RGB")

            teacher = read_teacher_labels(txt_path)
            total_boxes = len(teacher)
            if total_boxes == 0:
                if WRITE_EMPTY_ON_ALL_DROPPED: (out_labels / f"{stem}.txt").write_text("")
                continue

            
            metas = []
            crop_imgs: List[Image.Image] = []
            for (cls, xc, yc, w, h, conf) in teacher:
                crop, box, (W,H) = crop_from_yolo(image, xc, yc, w, h)
                tiny = (box[2]-box[0] < SMALL_BOX_MIN_PIX) or (box[3]-box[1] < SMALL_BOX_MIN_PIX)
                metas.append([cls, xc, yc, w, h, conf, box, tiny, (W,H)])
                if not tiny: crop_imgs.append(crop)

            
            crop_feats = torch.empty(0, class_embeds.size(1), device=DEVICE)
            if crop_imgs:
                crop_feats = rsclip.encode_images_batched(crop_imgs, INITIAL_BATCH_SIZE)

           
            per_box = [] 
            feat_ptr = 0
            for idx, (cls, xc, yc, w, h, conf, box, tiny, (W,H)) in enumerate(metas):
                info = {
                    "idx": idx, "cls": cls, "xc": xc, "yc": yc, "w": w, "h": h,
                    "conf": conf, "box": box, "tiny": tiny,
                    "tgt": None, "best": None, "best_idx": None, "margin": None,
                    "phase": "crop", "candidate": False, "red": False, "amber": False,
                    "final": "keep", "reason": "keep"
                }

                
                if tiny or (conf is not None and conf >= TEACHER_CONF_KEEP):
                    per_box.append(info)
                    continue

                v = crop_feats[feat_ptr] if not tiny else None
                if not tiny: feat_ptr += 1
                logits = (v @ class_embeds.T) * rsclip.logit_scale if v is not None else None
                tgt, best_val, best_idx, margin = first_pass_metrics(cls, logits) if logits is not None else (None,None,None,None)
                info.update({"tgt": tgt, "best": best_val, "best_idx": best_idx, "margin": margin})

                
                if (best_idx == cls) or (tgt is not None and tgt >= KEEP_MIN_PROB):
                    per_box.append(info)
                    continue

                
                info["candidate"] = True
                per_box.append(info)

            # Context pass for candidates
            cand_indices = [b["idx"] for b in per_box if b["candidate"]]
            if SECOND_PASS_CONTEXT and cand_indices:
                rechecked += len(cand_indices)
                ctx_imgs = [draw_box(image, metas[i][6], color="red", width=DRAW_WIDTH) for i in cand_indices]
                ctx_feats = rsclip.encode_images_batched(ctx_imgs, INITIAL_BATCH_SIZE)
                for j, bidx in enumerate(cand_indices):
                    b = per_box[bidx]
                    cls = b["cls"]
                    # fuse
                    crop_vec = crop_feats[[i for i,bb in enumerate(per_box) if bb["idx"]==bidx and bb["tgt"] is not None][0]]
                    ctx_vec  = ctx_feats[j]
                    fused = (CONTEXT_WEIGHT * ctx_vec + (1.0 - CONTEXT_WEIGHT) * crop_vec)
                    fused = fused / fused.norm(dim=-1, keepdim=True)
                    logits = (fused @ class_embeds.T) * rsclip.logit_scale
                    tgt, best_val, best_idx, margin = first_pass_metrics(cls, logits)
                    b.update({"tgt": tgt, "best": best_val, "best_idx": best_idx, "margin": margin, "phase": "context"})
                    # classify risk
                    b["red"]   = is_red_drop(tgt, margin, best_idx, cls)
                    b["amber"] = (USE_AMBER_BUCKET and is_amber_drop(tgt, margin, best_idx, cls) and not b["red"])

            
            
            red_list = [b for b in per_box if b["red"]]
            amber_list = [b for b in per_box if (not b["red"]) and b["amber"]]
            # rank amber by risk: lower tgt, higher margin first
            amber_list.sort(key=lambda b: (b["tgt"], -b["margin"]))
            keep_list = [b for b in per_box if not b["red"] and not b["amber"]]

            per_image_cap = int(MAX_DROP_FRACTION_PER_IMAGE * total_boxes)
            drops_assigned = 0

            
            for b in red_list:
                if drops_assigned < per_image_cap:
                    b["final"] = "drop"; b["reason"] = "red"; drops_assigned += 1
                else:
                    b["final"] = "keep"; b["reason"] = "cap_kept"

            # Drop ambers 
            need_more = max(0, TARGET_DROPS - dropped_boxes - drops_assigned) if TARGET_DROPS > 0 else 0
            amber_budget = max(0, min(per_image_cap - drops_assigned, need_more))
            for k, b in enumerate(amber_list):
                if k < amber_budget:
                    b["final"] = "drop"; b["reason"] = "amber"; drops_assigned += 1
                else:
                    b["final"] = "keep"; b["reason"] = "amber_kept"

            # Build output labels and audit
            new_lines = []
            dropped_here = 0
            for b in per_box:
                cls, xc, yc, w, h, conf = b["cls"], b["xc"], b["yc"], b["w"], b["h"], b["conf"]
                final = b["final"]
                if final == "keep":
                    new_lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                    kept_boxes += 1
                else:
                    dropped_here += 1

                audit_rows.append({
                    "image_stem": stem,
                    "cls": cls,
                    "label": id2name[cls],
                    "teacher_conf": "" if conf is None else f"{conf:.4f}",
                    "phase": b["phase"],
                    "tgt_prob": "" if b["tgt"] is None else f"{b['tgt']:.4f}",
                    "best_label": "" if b["best_idx"] is None else id2name.get(int(b["best_idx"]), str(b["best_idx"])),
                    "best_prob": "" if b["best"] is None else f"{b['best']:.4f}",
                    "margin": "" if b["margin"] is None else f"{b['margin']:.4f}",
                    "final_decision": final,
                    "reason": b["reason"],
                    "tiny": "1" if b["tiny"] else "0"
                })

            dropped_boxes += dropped_here

            # Write labels 
            out_path = out_labels / f"{stem}.txt"
            if new_lines:
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(new_lines))
            elif WRITE_EMPTY_ON_ALL_DROPPED:
                out_path.write_text("")

            # Debug images
            if out_boxed and saved_boxes < SAVE_BOXED_EXAMPLES:
                W,H = image.size
                kept_set = set()
                keep_boxes = []
                for line in new_lines:
                    parts = line.strip().split()
                    kcls = int(parts[0]); kxc,kyc,kw,kh = map(float, parts[1:5])
                    x1 = (kxc-kw/2)*W; y1=(kyc-kh/2)*H; x2=(kxc+kw/2)*W; y2=(kyc+kh/2)*H
                    keep_boxes.append((x1,y1,x2,y2))
                    kept_set.add((kcls, round(kxc,6), round(kyc,6), round(kw,6), round(kh,6)))
                drop_boxes = []
                for m in metas:
                    cls, xc, yc, w, h, conf, box, tiny, _ = m
                    key = (cls, round(xc,6), round(yc,6), round(w,6), round(h,6))
                    if key not in kept_set:
                        drop_boxes.append(tuple(box))
                vis = image.copy(); d = ImageDraw.Draw(vis)
                for b in keep_boxes: d.rectangle(b, outline="green", width=3)
                for b in drop_boxes: d.rectangle(b, outline="red", width=3)
                vis.save(out_boxed / f"{stem}.jpg"); saved_boxes += 1

            if SAVE_JSON:
                
                audit_blob[stem] = [{"cls": b["cls"], "final": b["final"], "reason": b["reason"]} for b in per_box]

    
    fieldnames = ["image_stem","cls","label","teacher_conf","phase","tgt_prob","best_label","best_prob","margin","final_decision","reason","tiny"]
    with open(out_audit, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(audit_rows)

    if SAVE_JSON:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(audit_blob, f, indent=2)

    print(f"=== DONE ===", flush=True)
    print(f"Kept boxes:   {kept_boxes}", flush=True)
    print(f"Dropped boxes:{dropped_boxes}", flush=True)
    print(f"Rechecked with context: {rechecked}", flush=True)
    print(f"Labels → {OUTDIR/'labels_verified'}", flush=True)
    print(f"Audit  → {OUTDIR/'audit.csv'}", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("FATAL ERROR:", repr(e), flush=True)
        traceback.print_exc()
        sys.exit(1)
