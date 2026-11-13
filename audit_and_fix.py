import os
import yaml
from pathlib import Path
from PIL import Image

BASE_DIR = Path(r"C:\Users\User\Documents\training-plant")
FINAL_DATASET_DIR = BASE_DIR / "phytoguard_dataset"
RAW_DATASETS = [
    BASE_DIR / "Potato-Disease--1",
    BASE_DIR / "Rice-Leaf-Disease-2",
    BASE_DIR / "Sugarcane-Leaf-Disease-Dataset-4",
    BASE_DIR / "tomato-leaf-disease-18",
    BASE_DIR / "Wheat-Disease-3"
]

CANONICAL_CLASSES = [
    "Potato___Early_blight", "Potato___Late_blight",
    "Wheat___Black_Rust", "Wheat___Brown_Rust", "Wheat___Healthy_Wheat", "Wheat___Yellow_Rust",
    "Sugarcane___yellow", "Sugarcane___Healthy", "Sugarcane___Rust", "Sugarcane___RedRot",
    "Rice___Bacterial_blight", "Rice___Blast", "Rice___Brown_spot", "Rice___Tungro",
    "Tomato___healthy", "Tomato___bacterial_spot", "Tomato___leaf_blight", "Tomato___leaf_mold"
]

LABEL_MAP = {
    "potato early blight": "Potato___Early_blight", "early blight": "Potato___Early_blight",
    "potato late blight": "Potato___Late_blight", "late blight": "Potato___Late_blight",
    "black rust": "Wheat___Black_Rust", "stem rust": "Wheat___Black_Rust",
    "brown rust": "Wheat___Brown_Rust", "leaf rust": "Wheat___Brown_Rust",
    "healthy wheat": "Wheat___Healthy_Wheat", "wheat healthy": "Wheat___Healthy_Wheat",
    "yellow rust": "Wheat___Yellow_Rust", "stripe rust": "Wheat___Yellow_Rust",
    "yellow": "Sugarcane___yellow", "sugarcane healthy": "Sugarcane___Healthy",
    "rust": "Sugarcane___Rust", "redrot": "Sugarcane___RedRot", "red rot": "Sugarcane___RedRot",
    "bacterial blight": "Rice___Bacterial_blight", "bacterial leaf blight": "Rice___Bacterial_blight",
    "blast": "Rice___Blast", "rice blast": "Rice___Blast",
    "brown spot": "Rice___Brown_spot", "rice brown spot": "Rice___Brown_spot",
    "tungro": "Rice___Tungro", "rice tungro": "Rice___Tungro",
    "tomato healthy": "Tomato___healthy", "bacterial spot": "Tomato___bacterial_spot",
    "leaf blight": "Tomato___leaf_blight", "tomato leaf blight": "Tomato___leaf_blight",
    "leaf mold": "Tomato___leaf_mold", "tomato leaf mold": "Tomato___leaf_mold"
}

def rescue_polygons_and_empty_crops():
    rescued = 0
    for dataset_path in RAW_DATASETS:
        yaml_path = dataset_path / "data.yaml"
        if not yaml_path.exists():
            continue
        with open(yaml_path, 'r') as f:
            class_names = yaml.safe_load(f).get('names', [])

        crop_context = ""
        fl = dataset_path.name.lower()
        if "potato" in fl: crop_context = "Potato___"
        elif "rice" in fl: crop_context = "Rice___"
        elif "sugarcane" in fl: crop_context = "Sugarcane___"
        elif "tomato" in fl: crop_context = "Tomato___"
        elif "wheat" in fl: crop_context = "Wheat___"

        for split in ["train", "valid", "test"]:
            img_dir = dataset_path / split / "images"
            lbl_dir = dataset_path / split / "labels"
            target_split = "val" if split == "valid" else split
            if not img_dir.exists():
                continue

            for img_file in img_dir.iterdir():
                if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                    continue
                lbl_file = lbl_dir / f"{img_file.stem}.txt"
                if not lbl_file.exists():
                    continue

                with open(lbl_file, 'r') as lf:
                    lines = lf.readlines()

                for i, line in enumerate(lines):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    class_id = int(parts[0])
                    raw_name = class_names[class_id].lower()
                    target_class = LABEL_MAP.get(raw_name)
                    if not target_class and crop_context:
                        cand = f"{crop_context}{raw_name.replace(' ', '_')}"
                        for c in CANONICAL_CLASSES:
                            if c.lower() == cand.lower():
                                target_class = c
                                break
                    if not target_class:
                        continue

                    save_path = FINAL_DATASET_DIR / target_split / target_class / f"{img_file.stem}_crop{i}.jpg"
                    if save_path.exists() and save_path.stat().st_size > 0:
                        continue  # Already saved cleanly

                    with Image.open(img_file) as img:
                        img = img.convert("RGB")
                        w, h = img.size
                        coords = list(map(float, parts[1:]))

                        # Handle Polygon annotations (>4 coords) vs BBox (4 coords)
                        if len(coords) > 4:
                            xs = coords[0::2]
                            ys = coords[1::2]
                            x1, x2 = int(min(xs) * w), int(max(xs) * w)
                            y1, y2 = int(min(ys) * h), int(max(ys) * h)
                        else:
                            xc, yc, bw, bh = coords[:4]
                            x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
                            x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)

                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)

                        # Fallback to full image if bounding box is degenerate (<10px)
                        if (x2 - x1) < 10 or (y2 - y1) < 10:
                            cropped = img
                        else:
                            cropped = img.crop((x1, y1, x2, y2))

                        cropped.save(save_path)
                        rescued += 1

    print(f"[RESCUED] Recovered {rescued} polygon/degenerate-box images!\n")

def audit_counts():
    print(f"{'CLASS NAME':<35} | {'TRAIN':>6} | {'VAL':>5} | {'TEST':>5} | STATUS")
    print("-" * 72)
    for cls in CANONICAL_CLASSES:
        tr = len(list((FINAL_DATASET_DIR / "train" / cls).glob("*.jpg")))
        va = len(list((FINAL_DATASET_DIR / "val" / cls).glob("*.jpg")))
        te = len(list((FINAL_DATASET_DIR / "test" / cls).glob("*.jpg")))
        status = "OK" if tr > 0 else "EMPTY! (Check data.yaml)"
        print(f"{cls:<35} | {tr:>6} | {va:>5} | {te:>5} | {status}")

if __name__ == "__main__":
    rescue_polygons_and_empty_crops()
    audit_counts()