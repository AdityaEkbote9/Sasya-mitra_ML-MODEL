import os
import shutil
import random
from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

BASE_DIR = Path(r"C:\Users\User\Documents\training-plant")
SOURCE_DIR = BASE_DIR / "phytoguard_dataset"
EXACT_1K_DIR = BASE_DIR / "phytoguard_1k"

# EXACT TARGETS
TARGET_TRAIN = 1000
MAX_EVAL = 100  # Cap val/test to 100 for a perfectly even evaluation grid

random.seed(42)

def apply_field_augmentation(src_path, dst_path):
    """Applies heavy real-world camera noise to oversample tiny classes."""
    try:
        with Image.open(src_path) as im:
            aug = im.copy().convert("RGB")
            
            if random.random() > 0.5:
                aug = ImageOps.mirror(aug)
            if random.random() > 0.5:
                aug = ImageOps.flip(aug)
                
            rot_angle = random.choice([90, 180, 270, random.uniform(-30, 30)])
            aug = aug.rotate(rot_angle, expand=True)
            
            if random.random() > 0.3:
                enhancer = ImageEnhance.Brightness(aug)
                aug = enhancer.enhance(random.uniform(0.6, 1.4))
                
            if random.random() > 0.3:
                enhancer = ImageEnhance.Contrast(aug)
                aug = enhancer.enhance(random.uniform(0.7, 1.3))
                
            if random.random() > 0.3:
                enhancer = ImageEnhance.Color(aug)
                aug = enhancer.enhance(random.uniform(0.5, 1.5))
                
            if random.random() > 0.7:
                aug = aug.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
                
            aug.save(dst_path)
    except Exception as e:
        print(f"Skipping corrupt image for augmentation: {src_path.name}")

def process_eval_split(split_name, max_cap):
    """Copies val/test data with a hard cap, NO augmentation to keep metrics honest."""
    src_split_dir = SOURCE_DIR / split_name
    dst_split_dir = EXACT_1K_DIR / split_name

    for cls_dir in sorted(src_split_dir.iterdir()):
        if not cls_dir.is_dir(): continue
        cls_name = cls_dir.name
        dst_cls_dir = dst_split_dir / cls_name
        dst_cls_dir.mkdir(parents=True, exist_ok=True)
        
        imgs = list(cls_dir.glob("*.jpg"))
        if not imgs: continue
        
        random.shuffle(imgs)
        selected = imgs[:max_cap]
        
        for idx, img_path in enumerate(selected):
            shutil.copy2(img_path, dst_cls_dir / f"{cls_name}_{idx}.jpg")

def process_train_split():
    """Forces every class to have EXACTLY TARGET_TRAIN (1000) images."""
    src_split_dir = SOURCE_DIR / "train"
    dst_split_dir = EXACT_1K_DIR / "train"

    for cls_dir in sorted(src_split_dir.iterdir()):
        if not cls_dir.is_dir(): continue
        cls_name = cls_dir.name
        dst_cls_dir = dst_split_dir / cls_name
        dst_cls_dir.mkdir(parents=True, exist_ok=True)
        
        imgs = list(cls_dir.glob("*.jpg"))
        if not imgs: continue
        
        random.shuffle(imgs)
        
        # 1. Downsample if class has > 1000 images
        if len(imgs) >= TARGET_TRAIN:
            for idx, img_path in enumerate(imgs[:TARGET_TRAIN]):
                shutil.copy2(img_path, dst_cls_dir / f"{cls_name}_{idx}.jpg")
                
        # 2. Augment if class has < 1000 images
        else:
            for idx, img_path in enumerate(imgs):
                shutil.copy2(img_path, dst_cls_dir / f"{cls_name}_{idx}.jpg")
            
            needed = TARGET_TRAIN - len(imgs)
            for i in range(needed):
                src_img = imgs[i % len(imgs)]
                dst_name = dst_cls_dir / f"{cls_name}_aug_{i}.jpg"
                apply_field_augmentation(src_img, dst_name)

def run():
    print("[INFO] Building exactly 1K balanced dataset...")
    if EXACT_1K_DIR.exists():
        shutil.rmtree(EXACT_1K_DIR)
        
    process_train_split()
    process_eval_split("val", MAX_EVAL)
    process_eval_split("test", MAX_EVAL)

    print("\n=== 1K EXACT BALANCED SUMMARY ===")
    print(f"{'CLASS NAME':<35} | {'TRAIN':>6} | {'VAL':>5} | {'TEST':>5}")
    print("-" * 62)
    
    for cls_dir in sorted((EXACT_1K_DIR / "train").iterdir()):
        tr = len(list((EXACT_1K_DIR / "train" / cls_dir.name).glob("*.jpg")))
        va = len(list((EXACT_1K_DIR / "val" / cls_dir.name).glob("*.jpg")))
        te = len(list((EXACT_1K_DIR / "test" / cls_dir.name).glob("*.jpg")))
        print(f"{cls_dir.name:<35} | {tr:>6} | {va:>5} | {te:>5}")

if __name__ == "__main__":
    run()