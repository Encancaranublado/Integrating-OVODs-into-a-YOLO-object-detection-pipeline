#!/usr/bin/env python3


from __future__ import annotations
import csv, json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from PIL import Image, ImageDraw
import yaml
from transformers import Owlv2Processor, Owlv2ForObjectDetection


TEACHER_LABELS_DIR = Path(r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\artifacts_student\yolo_train\labels")
TRAIN_IMAGES       = Path(r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\dataset\dataset_yolo_tiled\images\train")
NAMES_YAML         = Path(r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\dataset\dataset_yolo_tiled\data.yaml")
OUTDIR             = Path(r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\artifacts_student\owlv2_visualgrounding_artifacts\owlv2_grounded_train")

MODEL_ID = "google/owlv2-base-patch16-ensemble"
DEVICE   = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if ("cuda" in DEVICE and torch.cuda.is_bf16_supported()) else torch.float32

# Thresholds
OWL_SCORE_THR = 0.15
IOU_MATCH_THR = 0.50
REFINEMENT_MODE = "snap_to_owl"   
FUSE_ALPHA = 0.5
ADD_OWL_ONLY = False
ADD_OWL_MIN_SCORE = 0.40
MAX_TEXT_LABELS_PER_IMAGE = 80

# fail-safe mode
FAILSAFE_KEEP = True  

# Visualizations
VIS = True
VIS_EVERY = 100


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_names(names_yaml: Path) -> Dict[int,str]:
    with open(names_yaml,"r",encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names")
    if isinstance(names, dict):
        return {int(k): str(v) for k,v in names.items()}
    elif isinstance(names, list):
        return {i: str(n) for i,n in enumerate(names)}
    else:
        raise ValueError("Bad data.yaml format")

def img_paths_by_stem(root: Path) -> Dict[str,Path]:
    exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp",".JPG",".PNG",".TIF",".WEBP"}
    return {p.stem:p for p in root.rglob("*") if p.suffix in exts}

def read_teacher_labels(txt_path: Path):
    rows=[]
    with open(txt_path,"r",encoding="utf-8") as f:
        for line in f:
            parts=line.strip().split()
            if len(parts)<5: continue
            cls=int(float(parts[0]))
            xc,yc,w,h=map(float,parts[1:5])
            conf=float(parts[5]) if len(parts)>=6 else None
            rows.append((cls,xc,yc,w,h,conf))
    return rows

def yolo_norm_to_xyxy(xc,yc,w,h,W,H):
    return ( (xc-w/2)*W, (yc-h/2)*H, (xc+w/2)*W, (yc+h/2)*H )

def xyxy_to_yolo_norm(x1,y1,x2,y2,W,H):
    w=(x2-x1)/W; h=(y2-y1)/H
    xc=(x1+x2)/(2*W); yc=(y1+y2)/(2*H)
    return xc,yc,w,h

def box_iou_xyxy(a, b, W=None, H=None):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    
    if W is not None and H is not None:
        ax1, ax2 = np.clip([ax1, ax2], 0, W)
        bx1, bx2 = np.clip([bx1, bx2], 0, W)
        ay1, ay2 = np.clip([ay1, ay2], 0, H)
        by1, by2 = np.clip([by1, by2], 0, H)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter + 1e-9
    return inter / union


def make_prompts_for_classes(class_ids,id2name):
    prompts=[]
    for cid in class_ids:
        name=id2name.get(cid,str(cid)).strip()
        prompts.append(f"a {name.lower()}")
    seen=set(); uniq=[]
    for p in prompts:
        if p not in seen:
            uniq.append(p); seen.add(p)
    return uniq[:MAX_TEXT_LABELS_PER_IMAGE]

def draw_boxes(img,boxes_xyxy,colors,labels):
    d=ImageDraw.Draw(img)
    for (x1,y1,x2,y2),c,l in zip(boxes_xyxy,colors,labels):
        d.rectangle([x1,y1,x2,y2],outline=c,width=2)
        d.text((x1+2,max(0,y1-12)),l,fill=c)
    return img


def main():
    ensure_dir(OUTDIR)
    out_labels=ensure_dir(OUTDIR/"labels_grounded")
    out_audit=OUTDIR/"audit.csv"; out_stats=OUTDIR/"stats.json"
    out_vis=ensure_dir(OUTDIR/"vis") if VIS else None

    id2name=load_names(NAMES_YAML)
    img_by_stem=img_paths_by_stem(TRAIN_IMAGES)

    print(f"[LOAD] {MODEL_ID} on {DEVICE}")
    processor=Owlv2Processor.from_pretrained(MODEL_ID)
    model=Owlv2ForObjectDetection.from_pretrained(MODEL_ID,torch_dtype=TORCH_DTYPE).to(DEVICE).eval()

    txt_files=sorted(TEACHER_LABELS_DIR.glob("*.txt"))
    kept_total=dropped_total=refined_total=added_owl_total=0
    rows_for_csv=[]

    for idx,txt_path in enumerate(txt_files):
        stem=txt_path.stem
        img_path=img_by_stem.get(stem)
        if img_path is None: continue
        image=Image.open(img_path).convert("RGB"); W,H=image.size
        teacher=read_teacher_labels(txt_path)
        if not teacher: continue

        class_ids=[t[0] for t in teacher]
        prompts=make_prompts_for_classes(class_ids,id2name)
        text_labels=[prompts]

        with torch.no_grad():
            inputs=processor(text=text_labels,images=image,return_tensors="pt").to(DEVICE)
            outputs=model(**inputs)
            target_sizes=torch.tensor([(H,W)],device=DEVICE)
            results=processor.post_process_grounded_object_detection(outputs, target_sizes=target_sizes,
                                threshold=OWL_SCORE_THR,text_labels=text_labels)
        result=results[0]
        owl_boxes=result["boxes"].cpu().numpy()
        owl_scores=result["scores"].cpu().numpy().tolist()
        owl_texts=[str(t) for t in result["text_labels"]]

        by_text={}
        for b,s,t in zip(owl_boxes,owl_scores,owl_texts):
            by_text.setdefault(t,[]).append((b,s))

        grounded_boxes_xyxy=[]; grounded_src=[]

        for (cls,xc,yc,w,h,conf_t) in teacher:
            name=id2name.get(cls,str(cls))
            prompt=f"a {name.lower()}"
            teacher_xyxy=np.array(yolo_norm_to_xyxy(xc,yc,w,h,W,H))

            candidates=by_text.get(prompt,[])
            best_iou=0.0; best=None
            for b,s in candidates:
                iou=box_iou_xyxy(teacher_xyxy,b)
                if iou>best_iou: best_iou=iou; best=(b,s)

            if best is not None and best_iou>=IOU_MATCH_THR:
                owl_b,owl_s=best; kept_total+=1
                if REFINEMENT_MODE=="snap_to_owl":
                    x1,y1,x2,y2=owl_b.tolist(); grounded_src.append("owl")
                elif REFINEMENT_MODE=="fuse":
                    x1o,y1o,x2o,y2o=owl_b.tolist(); x1t,y1t,x2t,y2t=teacher_xyxy.tolist()
                    x1=FUSE_ALPHA*x1t+(1-FUSE_ALPHA)*x1o; y1=FUSE_ALPHA*y1t+(1-FUSE_ALPHA)*y1o
                    x2=FUSE_ALPHA*x2t+(1-FUSE_ALPHA)*x2o; y2=FUSE_ALPHA*y2t+(1-FUSE_ALPHA)*y2o
                    grounded_src.append("fused")
                else:
                    x1,y1,x2,y2=teacher_xyxy.tolist(); grounded_src.append("teacher")
                grounded_boxes_xyxy.append((cls,x1,y1,x2,y2)); refined_total+=1
                rows_for_csv.append({"image_stem":stem,"decision":"keep","cls":cls,
                    "teacher_conf":conf_t if conf_t else "","owl_score":owl_s,
                    "iou_owl_teacher":best_iou,"source_used":grounded_src[-1]})
            else:
                if FAILSAFE_KEEP:
                    kept_total+=1
                    x1,y1,x2,y2=teacher_xyxy.tolist()
                    grounded_boxes_xyxy.append((cls,x1,y1,x2,y2))
                    grounded_src.append("teacher_failsafe")
                    rows_for_csv.append({"image_stem":stem,"decision":"keep_failsafe","cls":cls,
                        "teacher_conf":conf_t if conf_t else "","owl_score":"",
                        "iou_owl_teacher":best_iou,"source_used":"teacher_failsafe"})
                else:
                    dropped_total+=1
                    rows_for_csv.append({"image_stem":stem,"decision":"drop","cls":cls,
                        "teacher_conf":conf_t if conf_t else "","owl_score":"",
                        "iou_owl_teacher":best_iou,"source_used":""})

        
        with open(out_labels/f"{stem}.txt","w",encoding="utf-8") as f:
            for (cls,x1,y1,x2,y2) in grounded_boxes_xyxy:
                xc,yc,wn,hn=xyxy_to_yolo_norm(x1,y1,x2,y2,W,H)
                f.write(f"{cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

        
        if VIS and (VIS_EVERY==0 or idx%VIS_EVERY==0):
            vis=image.copy()
            teacher_xy=[yolo_norm_to_xyxy(t[1],t[2],t[3],t[4],W,H) for t in teacher]
            vis=draw_boxes(vis,teacher_xy,["red"]*len(teacher_xy),["T"]*len(teacher_xy))
            vis=draw_boxes(vis,[(b[1],b[2],b[3],b[4]) for b in grounded_boxes_xyxy],
                           ["green"]*len(grounded_boxes_xyxy),["G"]*len(grounded_boxes_xyxy))
            vis.save(out_vis/f"{stem}.jpg")

        if (idx+1)%100==0:
            print(f"[PROGRESS] {idx+1}/{len(txt_files)} images | kept={kept_total} dropped={dropped_total} refined={refined_total} added_owl={added_owl_total}")

    
    with open(out_audit,"w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=["image_stem","decision","cls","teacher_conf","owl_score","iou_owl_teacher","source_used"])
        writer.writeheader(); writer.writerows(rows_for_csv)
    stats={"kept_total":kept_total,"dropped_total":dropped_total,"refined_total":refined_total,
           "added_owl_total":added_owl_total,"images":len(txt_files),"model":MODEL_ID,
           "OWL_SCORE_THR":OWL_SCORE_THR,"IOU_MATCH_THR":IOU_MATCH_THR,"REFINEMENT_MODE":REFINEMENT_MODE,
           "ADD_OWL_ONLY":ADD_OWL_ONLY,"FAILSAFE_KEEP":FAILSAFE_KEEP}
    with open(out_stats,"w",encoding="utf-8") as f: json.dump(stats,f,indent=2)
    print("[DONE] Grounding complete.")

if __name__=="__main__":
    main()
