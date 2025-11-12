#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import csv, json
from collections import Counter, defaultdict
import yaml


TEACHER_LABELS_DIR = Path(r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\artifacts_student\yolo_train\labels")
RSCLIP_LABELS_DIR  = Path(r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\artifacts_student\rsclip_verified_train_fast\labels_verified")
RSCLIP_AUDIT_CSV   = Path(r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\artifacts_student\rsclip_verified_train_fast\audit.csv")  
OWL_AUDIT_CSV      = Path(r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\artifacts_student\rsclip_owlv2_refined\audit.csv")         
NAMES_YAML         = Path(r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\dataset\dataset_yolo_tiled\data.yaml")  

OUTDIR             = Path(r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\artifacts_student\metrics_summary")



def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def load_names(yaml_path: Path) -> dict[int,str]:
    if not yaml_path or not yaml_path.exists(): return {}
    d = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = d.get("names")
    if isinstance(names, dict):
        return {int(k): str(v) for k,v in names.items()}
    if isinstance(names, list):
        return {i: str(n) for i,n in enumerate(names)}
    return {}

def parse_yolo_counts(lbl_dir: Path) -> Counter:
    cnt = Counter()
    if not lbl_dir.exists(): return cnt
    for p in lbl_dir.glob("*.txt"):
        txt = p.read_text(encoding="utf-8")
        for ln in txt.splitlines():
            parts = ln.strip().split()
            if len(parts) >= 5:
                try:
                    cid = int(float(parts[0]))
                    cnt[cid] += 1
                except Exception:
                    pass
    return cnt

def clip_drops_from_audit(audit_csv: Path) -> tuple[int, Counter] | None:
    if not audit_csv.exists(): return None
    with audit_csv.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if "final_decision" not in r.fieldnames:
            
            return None
        dropped = 0
        per_class = Counter()
        for row in r:
            if row.get("final_decision","").strip().lower() == "drop":
                dropped += 1
                # robust class id parse
                cls_raw = row.get("cls", "")
                try:
                    cid = int(float(cls_raw))
                    per_class[cid] += 1
                except Exception:
                    pass
    return (dropped, per_class)

def clip_drops_by_diff(teacher_dir: Path, rsclip_dir: Path) -> tuple[int, Counter]:
    t_cnt = parse_yolo_counts(teacher_dir)
    f_cnt = parse_yolo_counts(rsclip_dir)
    
    per_class_drop = Counter()
    class_ids = set(t_cnt.keys()) | set(f_cnt.keys())
    for cid in class_ids:
        d = t_cnt.get(cid,0) - f_cnt.get(cid,0)
        if d > 0:
            per_class_drop[cid] = d
    total_drop = sum(per_class_drop.values())
    return total_drop, per_class_drop

def owl_refinements_from_audit(audit_csv: Path) -> tuple[int,int,int, Counter]:
    refined = 0
    failsafe = 0
    dropped = 0
    per_class_ref = Counter()

    if not audit_csv.exists(): return (0,0,0,per_class_ref)

    with audit_csv.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fields = set(r.fieldnames or [])
        src_field = "src" if "src" in fields else ("source_used" if "source_used" in fields else None)
        dec_field = "decision" if "decision" in fields else None

        for row in r:
            # class id
            cid = None
            try:
                cid = int(float(row.get("cls","")))
            except Exception:
                pass

            decision = (row.get(dec_field,"") if dec_field else "").strip().lower()
            src = (row.get(src_field,"") if src_field else "").strip().lower()

            is_ref = (decision == "keep_refined") or (src in {"owl","fused"})
            is_failsafe = (decision == "keep_failsafe") or (src == "teacher_failsafe")
            is_drop = (decision == "drop")

            if is_ref:
                refined += 1
                if cid is not None:
                    per_class_ref[cid] += 1
            elif is_failsafe:
                failsafe += 1
            elif is_drop:
                dropped += 1
            else:
                
                pass

    return refined, failsafe, dropped, per_class_ref

def write_per_class_csv(path: Path, counts: Counter, id2name: dict[int,str]):
    rows = []
    for cid, n in sorted(counts.items()):
        rows.append({
            "class_id": cid,
            "class_name": id2name.get(cid, "UNKNOWN"),
            "count": n
        })
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["class_id","class_name","count"])
        w.writeheader(); w.writerows(rows)

def main():
    ensure_dir(OUTDIR)
    id2name = load_names(NAMES_YAML)

    
    clip_method = "audit"
    clip_res = clip_drops_from_audit(RSCLIP_AUDIT_CSV)
    if clip_res is None:
        clip_method = "diff_teacher_vs_filtered"
        dropped_total, per_class_clip = clip_drops_by_diff(TEACHER_LABELS_DIR, RSCLIP_LABELS_DIR)
    else:
        dropped_total, per_class_clip = clip_res

    
    refined_total, failsafe_total, owl_dropped_total, per_class_owl = owl_refinements_from_audit(OWL_AUDIT_CSV)

    
    summary = {
        "clip": {
            "dropped_total": dropped_total,
            "method": clip_method,
        },
        "owl": {
            "refined_total": refined_total,
            "failsafe_kept": failsafe_total,
            "dropped_total": owl_dropped_total,
        }
    }
    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (OUTDIR / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric_group","metric","value","notes"])
        w.writerow(["CLIP","dropped_total", dropped_total, clip_method])
        w.writerow(["OWL","refined_total", refined_total, "refined := keep_refined or src in {owl,fused}"])
        w.writerow(["OWL","failsafe_kept", failsafe_total, "kept due to FAILSAFE_KEEP"])
        w.writerow(["OWL","dropped_total", owl_dropped_total, "strict OWL miss (not kept)"])

    
    if per_class_clip:
        write_per_class_csv(OUTDIR / "per_class_clip.csv", per_class_clip, id2name)
    if per_class_owl:
        write_per_class_csv(OUTDIR / "per_class_owl.csv", per_class_owl, id2name)

    
    print("\n=== SUMMARY ===")
    print(f"CLIP: dropped_total = {dropped_total}   (method: {clip_method})")
    print(f"OWL : refined_total = {refined_total} | failsafe_kept = {failsafe_total} | dropped_total = {owl_dropped_total}")
    if per_class_clip:
        print(f"Per-class CLIP drops saved → {OUTDIR/'per_class_clip.csv'}")
    if per_class_owl:
        print(f"Per-class OWL refinements saved → {OUTDIR/'per_class_owl.csv'}")
    print(f"JSON/CSV summaries → {OUTDIR/'summary.json'} , {OUTDIR/'summary.csv'}")

if __name__ == "__main__":
    main()
