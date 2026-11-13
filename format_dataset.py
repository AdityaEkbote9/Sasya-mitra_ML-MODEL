import os
import shutil
import yaml
from pathlib import Path
from PIL import Image

# 1. Your exact 5 downloaded folders from the VS Code workspace
BASE_DIR = Path(r"C:\Users\User\Documents\training-plant")
RAW_DATASETS = [
    BASE_DIR / "Potato-Disease--1",
    BASE_DIR / "Rice-Leaf-Disease-2",
    BASE_DIR / "Sugarcane-Leaf-Disease-Dataset-4",
    BASE_DIR / "tomato-leaf-disease-18",
    BASE_DIR / "Wheat-Disease-3"
]

FINAL_DATASET_DIR = BASE_DIR / "phytoguard_dataset"

# 2. UPDATED Canonical 18 Classes
CANONICAL_CLASSES = [
    "Potato___Early_blight", "Potato___Late_blight",
    "Wheat___Black_Rust", "Wheat___Brown_Rust", "Wheat___Healthy_Wheat", "Wheat___Yellow_Rust",
    "Sugarcane___yellow", "Sugarcane___Healthy", "Sugarcane___Rust", "Sugarcane___RedRot",
    "Rice___Bacterial_blight", "Rice___Blast", "Rice___Brown_spot", "Rice___Tungro",
    "Tomato___healthy", "Tomato___bacterial_spot", "Tomato___leaf_blight", "Tomato___leaf_mold"
]

# 3. Label Map (Matches raw Roboflow yaml class names to the new 18 canonical classes)
LABEL_MAP = {
    # Potato
    "potato early blight": "Potato___Early_blight", "early blight": "Potato___Early_blight",
    "potato late blight": "Potato___Late_blight", "late blight": "Potato___Late_blight",
    
    # Wheat
    "black rust": "Wheat___Black_Rust", "stem rust": "Wheat___Black_Rust",
    "brown rust": "Wheat___Brown_Rust", "leaf rust": "Wheat___Brown_Rust",
    "healthy wheat": "Wheat___Healthy_Wheat", "wheat healthy": "Wheat___Healthy_Wheat", 
    "healthy": "Wheat___Healthy_Wheat", 
    "yellow rust": "Wheat___Yellow_Rust", "stripe rust": "Wheat___Yellow_Rust",
    
    # Sugarcane
    "yellow": "Sugarcane___yellow", 
    "sugarcane healthy": "Sugarcane___Healthy", "healthy": "Sugarcane___Healthy",
    "rust": "Sugarcane___Rust", 
    "redrot": "Sugarcane___RedRot", "red rot": "Sugarcane___RedRot",
    
    # Rice
    "bacterial blight": "Rice___Bacterial_blight", "bacterial leaf blight": "Rice___Bacterial_blight",
    "blast": "Rice___Blast", "rice blast": "Rice___Blast",
    "brown spot": "Rice___Brown_spot", "rice brown spot": "Rice___Brown_spot",
    "tungro": "Rice___Tungro", "rice tungro": "Rice___Tungro",
    
    # Tomato
    "tomato healthy": "Tomato___healthy", "healthy": "Tomato___healthy",
    "bacterial spot": "Tomato___bacterial_spot", 
    "leaf blight": "Tomato___leaf_blight", "tomato leaf blight": "Tomato___leaf_blight",
    "leaf mold": "Tomato___leaf_mold", "tomato leaf mold": "Tomato___leaf_mold"
}

def create_dirs():
    for split in ["train", "val", "test"]:
        for cls in CANONICAL_CLASSES:
            (FINAL_DATASET_DIR / split / cls).mkdir(parents=True, exist_ok=True)

def process_datasets():
    create_dirs()
    
    for dataset_path in RAW_DATASETS:
        if not dataset_path.exists():
            print(f"[SKIP] Not found: {dataset_path.name}")
            continue
            
        yaml_path = dataset_path / "data.yaml"
        if not yaml_path.exists():
            print(f"[SKIP] No data.yaml in {dataset_path.name}")
            continue
            
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            class_names = data.get('names', [])
            
        # Determine the crop context from the folder name
        crop_context = ""
        folder_lower = dataset_path.name.lower()
        if "potato" in folder_lower: crop_context = "Potato___"
        elif "rice" in folder_lower: crop_context = "Rice___"
        elif "sugarcane" in folder_lower: crop_context = "Sugarcane___"
        elif "tomato" in folder_lower: crop_context = "Tomato___"
        elif "wheat" in folder_lower: crop_context = "Wheat___"
            
        print(f"\nProcessing: {dataset_path.name} (Context: {crop_context})")
        
        for split in ["train", "valid", "test"]:
            split_dir = dataset_path / split
            if not split_dir.exists(): continue
                
            img_dir = split_dir / "images"
            lbl_dir = split_dir / "labels"
            
            target_split = "val" if split == "valid" else split
            
            if not img_dir.exists(): continue
                
            for img_file in img_dir.iterdir():
                if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']: continue
                
                lbl_file = lbl_dir / f"{img_file.stem}.txt"
                if not lbl_file.exists(): continue
                
                with open(lbl_file, 'r') as lf:
                    lines = lf.readlines()
                    
                if not lines: continue
                
                for i, line in enumerate(lines):
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    
                    raw_class_name = class_names[class_id].lower()
                    target_class = LABEL_MAP.get(raw_class_name)
                    
                    # Context fallback for generic names like "healthy" or "rust"
                    if not target_class and crop_context:
                        formatted_raw = raw_class_name.replace(" ", "_")
                        constructed_class = f"{crop_context}{formatted_raw}"
                        # Check case-insensitive match against canonical classes
                        for canon in CANONICAL_CLASSES:
                            if canon.lower() == constructed_class.lower():
                                target_class = canon
                                break
                            
                    if target_class:
                        try:
                            with Image.open(img_file) as img:
                                img_w, img_h = img.size
                                x1 = int((x_center - width / 2) * img_w)
                                y1 = int((y_center - height / 2) * img_h)
                                x2 = int((x_center + width / 2) * img_w)
                                y2 = int((y_center + height / 2) * img_h)
                                
                                cropped_img = img.crop((max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)))
                                
                                save_path = FINAL_DATASET_DIR / target_split / target_class / f"{img_file.stem}_crop{i}.jpg"
                                cropped_img.save(save_path)
                        except Exception as e:
                            print(f"  [ERROR] Failed to process {img_file.name}: {e}")

    print("\n[SUCCESS] 18-Class Dataset cleanly cropped and formatted!")

if __name__ == "__main__":
    process_datasets()