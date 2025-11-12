#!/usr/bin/env python3

import sys
import subprocess
from pathlib import Path

# ========== CONFIG ==========
DATA_YAML  = Path("C:/Users/elpob/Documents/Projects/new_pipeline/yolo_pipeline/dataset/dataset_yolo_tiled/data.yaml")

IMG_SIZE   = 1536         
BATCH_SIZE = -1           
EPOCHS     = 50
WEIGHTS    = "yolov5m.pt"   
DEVICE     = "0"            
SEED       = 42

PROJECT    = Path(__file__).parents[1] / "runs" / "train"
RUN_NAME   = "exp1"

YOLOV5_URL = "https://github.com/ultralytics/yolov5.git"

# ========== PATHS ==========
ROOT       = Path(__file__).resolve().parents[1]
YOLOV5_DIR = ROOT / "yolov5"
TRAIN_PY   = YOLOV5_DIR / "train.py"

def main():
    
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"data.yaml not found: {DATA_YAML}")

    
    if not YOLOV5_DIR.exists():
        print(f"[INFO] Cloning YOLOv5 into {YOLOV5_DIR}")
        subprocess.run(
            ["git", "clone", "--depth", "1", YOLOV5_URL, str(YOLOV5_DIR)],
            check=True
        )

    if not TRAIN_PY.exists():
        raise FileNotFoundError(f"train.py not found at {TRAIN_PY}")

    
    (PROJECT / RUN_NAME).mkdir(parents=True, exist_ok=True)

    
    cmd = [
        sys.executable, str(TRAIN_PY),
        "--data",   str(DATA_YAML),
        "--img",    str(IMG_SIZE),
        "--batch",  str(BATCH_SIZE),
        "--epochs", str(EPOCHS),
        "--weights",WEIGHTS,
        "--device", DEVICE,
        "--seed",   str(SEED),
        "--project",str(PROJECT),
        "--name",   RUN_NAME,
    ]

    print("[INFO] Launching YOLOv5 training:")
    print("      " + " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)
    print(f"[INFO] Training finished. Check {PROJECT / RUN_NAME} for results.")

if __name__ == "__main__":
    main()
