import shutil
import random
from pathlib import Path
from PIL import Image, ImageOps

BASE_DIR = Path(r"C:\Users\User\Documents\training-plant")
FINAL_DATASET_DIR = BASE_DIR / "phytoguard_dataset"

MAX_TRAIN_PER_CLASS = 1200
MIN_TRAIN_PER_CLASS = 250
random.seed(42)

def clean_and_balance():
    train_root = FINAL_DATASET_DIR / "train"
    
    # 1. Remove any class that has 0 training images across train, val, and test
    for cls_dir in sorted(train_root.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls_name = cls_dir.name
        train_imgs = list(cls_dir.glob("*.jpg"))
        
        if len(train_imgs) == 0:
            print(f"[DELETED EMPTY CLASS] {cls_name}")
            for split in ["train", "val", "test"]:
                target = FINAL_DATASET_DIR / split / cls_name
                if target.exists():
                    shutil.rmtree(target)

    # 2. Balance remaining non-empty classes
    print("\n=== BALANCED PHYTOGUARD DATASET SUMMARY ===")
    print(f"{'CLASS NAME':<35} | {'TRAIN':>6} | {'VAL':>5} | {'TEST':>5}")
    print("-" * 62)

    for cls_dir in sorted(train_root.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls_name = cls_dir.name
        imgs = list(cls_dir.glob("*.jpg"))

        # Cap oversized classes (e.g., 14,945 Sugarcane Rust -> 1,200)
        if len(imgs) > MAX_TRAIN_PER_CLASS:
            random.shuffle(imgs)
            for extra in imgs[MAX_TRAIN_PER_CLASS:]:
                extra.unlink()

        # Oversample tiny classes (e.g., Wheat Black/Brown/Yellow Rust -> 250)
        elif len(imgs) < MIN_TRAIN_PER_CLASS:
            orig_imgs = list(imgs)
            idx = 0
            while len(list(cls_dir.glob("*.jpg"))) < MIN_TRAIN_PER_CLASS:
                src = orig_imgs[idx % len(orig_imgs)]
                with Image.open(src) as im:
                    aug = im.copy()
                    if idx % 3 == 0:
                        aug = ImageOps.mirror(aug)
                    elif idx % 3 == 1:
                        aug = ImageOps.flip(aug)
                    else:
                        aug = aug.rotate(random.choice([90, 180, 270]), expand=True)
                    aug.save(cls_dir / f"{src.stem}_aug{idx}.jpg")
                idx += 1

        tr = len(list((FINAL_DATASET_DIR / "train" / cls_name).glob("*.jpg")))
        va = len(list((FINAL_DATASET_DIR / "val" / cls_name).glob("*.jpg")))
        te = len(list((FINAL_DATASET_DIR / "test" / cls_name).glob("*.jpg")))
        print(f"{cls_name:<35} | {tr:>6} | {va:>5} | {te:>5}")

if __name__ == "__main__":
    clean_and_balance()