#!/usr/bin/env python3

from __future__ import annotations
from pathlib import Path
import subprocess, sys, shutil, yaml, re


YOLOV5_DIR = Path(r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\yolov5")  
DATA_YAML  = Path(r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\artifacts_student\student_datasets_all\fp_filter_plus_refine\data_fp_filter_plus_refine.yaml")
WEIGHTS    = "yolov5m.pt"   


EPOCHS   = 25
IMG_SIZE = 1536
BATCH    = 2
WORKERS  = 8
PATIENCE = 25

PROJECT  = Path(r"C:\Users\elpob\Documents\Projects\new_pipeline\yolo_pipeline\artifacts_student\yolov5_runs")
NAME     = "student_fp_filter_plus_refine"


PREFS = {
    "cache": "ram",          
    "save_period": 5,        
    "optimizer": "SGD",      
    "cos_lr": True,          
    "label_smoothing": 0.0,  
    "warmup_epochs": 3,      
    "freeze": 0,             
}

# ======================
def must_exist(p: Path, desc: str):
    if not p.exists():
        raise FileNotFoundError(f"Missing {desc}: {p}")

def get_help_text(train_py: Path) -> str:
    cmd = [sys.executable, str(train_py), "-h"]
    out = subprocess.check_output(cmd, cwd=str(train_py.parent), stderr=subprocess.STDOUT)
    return out.decode("utf-8", errors="ignore")

def flag_supported(help_text: str, flag: str) -> bool:
    return (flag in help_text)

def cache_accepts_value(help_text: str) -> bool:
    m = re.search(r"--cache[^\n]*\{?images|ram|disk", help_text)
    return bool(m)

def build_cmd(help_text: str) -> list[str]:
    train_py = YOLOV5_DIR / "train.py"
    cmd = [
        sys.executable, str(train_py),
        "--img", str(IMG_SIZE),
        "--batch", str(BATCH),
        "--epochs", str(EPOCHS),
        "--data", str(DATA_YAML),
        "--weights", str(WEIGHTS),
        "--project", str(PROJECT),
        "--name", NAME,
        "--exist-ok",
        "--workers", str(WORKERS),
        "--patience", str(PATIENCE),
    ]

    
    device = "0" if shutil.which("nvidia-smi") else "cpu"
    cmd += ["--device", device]

    
    if flag_supported(help_text, "--cache"):
        if cache_accepts_value(help_text):
            
            if PREFS.get("cache") in ("ram", "disk", "images"):
                cmd += ["--cache", PREFS["cache"]]
        else:
            
            if PREFS.get("cache"):  
                cmd += ["--cache"]

    
    if PREFS.get("save_period", 0) and flag_supported(help_text, "--save-period"):
        cmd += ["--save-period", str(int(PREFS["save_period"]))]

    
    if PREFS.get("optimizer") and flag_supported(help_text, "--optimizer"):
        cmd += ["--optimizer", str(PREFS["optimizer"])]

    
    if PREFS.get("cos_lr") and flag_supported(help_text, "--cos-lr"):
        cmd += ["--cos-lr"]

    
    if PREFS.get("label_smoothing", 0) and flag_supported(help_text, "--label-smoothing"):
        cmd += ["--label-smoothing", str(float(PREFS["label_smoothing"]))]

    
    if PREFS.get("warmup_epochs", 0) and flag_supported(help_text, "--warmup-epochs"):
        cmd += ["--warmup-epochs", str(int(PREFS["warmup_epochs"]))]

    
    if PREFS.get("freeze", 0) and flag_supported(help_text, "--freeze"):
        cmd += ["--freeze", str(int(PREFS["freeze"]))]

    return cmd

def main():
    
    must_exist(YOLOV5_DIR, "YOLOv5 repo dir")
    must_exist(YOLOV5_DIR / "train.py", "YOLOv5 train.py")
    must_exist(DATA_YAML, "dataset YAML")

    
    PROJECT.mkdir(parents=True, exist_ok=True)

    
    d = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))
    if "train" not in d or "val" not in d:
        raise ValueError(f"YAML must define 'train' and 'val': {DATA_YAML}")

    
    help_text = get_help_text(YOLOV5_DIR / "train.py")
    cmd = build_cmd(help_text)

    print("\n[YOLOv5 train.py detected flags]")
    print("cos-lr:", flag_supported(help_text, "--cos-lr"),
          "| cache accepts value:", cache_accepts_value(help_text),
          "| optimizer:", flag_supported(help_text, "--optimizer"),
          "| save-period:", flag_supported(help_text, "--save-period"),
          "| label-smoothing:", flag_supported(help_text, "--label-smoothing"),
          "| warmup-epochs:", flag_supported(help_text, "--warmup-epochs"),
          "| freeze:", flag_supported(help_text, "--freeze"))
    print("\n[RUN]\n", " ".join(cmd), "\n")

    
    subprocess.run(cmd, check=True, cwd=str(YOLOV5_DIR))

    run_dir = PROJECT / NAME
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    print(f"\n[DONE] Training complete.")
    print(f"best.pt: {best}")
    print(f"last.pt: {last}")

if __name__ == "__main__":
    main()
