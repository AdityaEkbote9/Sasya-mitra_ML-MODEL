from pathlib import Path
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

DATASET_DIR = r"C:\Users\User\Documents\training-plant\phytoguard_1k"
PROJECT_DIR = r"C:\Users\User\Documents\training-plant\runs"

def evaluate_test_set(model_pt_path: str, test_dir: str):
    """Generates Precision, Recall, F1-Score, and Confusion Matrix on the held-out test set."""
    model = YOLO(model_pt_path)
    results = model.predict(source=f"{test_dir}/**/*", stream=True, verbose=False)

    probs_list, labels_list = [], []
    class_to_idx = {v: k for k, v in model.names.items()}

    for r in results:
        true_cls = Path(r.path).parent.name
        if true_cls in class_to_idx:
            probs_list.append(r.probs.data.cpu().numpy())
            labels_list.append(class_to_idx[true_cls])

    y_true = np.array(labels_list)
    y_pred = np.argmax(np.array(probs_list), axis=1)

    report = classification_report(y_true, y_pred, target_names=list(model.names.values()), digits=4)
    print("\n=== PHYTOGUARD 16-CLASS TEST EVALUATION REPORT ===\n", report)
    with open(r"C:\Users\User\Documents\training-plant\phytoguard_eval_report.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(13, 11))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=list(model.names.values()),
        yticklabels=list(model.names.values())
    )
    plt.title("PHYTOGUARD 16-Class Confusion Matrix")
    plt.tight_layout()
    plt.savefig(r"C:\Users\User\Documents\training-plant\phytoguard_confusion_matrix.png", dpi=300)
    print("[SAVED] Confusion Matrix -> phytoguard_confusion_matrix.png")

if __name__ == "__main__":
    # 1. Load YOLOv11 Medium Classification backbone
    model = YOLO("yolo11m-cls.pt")

    # 2. Train on your Windows RTX GPU
    model.train(
        data=DATASET_DIR,
        epochs=15,
        imgsz=224,
        batch=16,              
        device=0,               
        workers=4,              # Safe multiprocessing count for Windows
        optimizer="AdamW",
        lr0=0.001,
        patience=8,             # Early stopping if validation accuracy plateaus
        # Field-realism augmentations for live stage lighting & angles
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=20.0,
        flipud=0.5,
        fliplr=0.5,
        erasing=0.15,
        project=PROJECT_DIR,
        name="phytoguard_16cls"
    )

    best_weights = f"{PROJECT_DIR}/phytoguard_16cls/weights/best.pt"
    print(f"\n[SUCCESS] Production weights saved at: {best_weights}")

    # 3. Auto-generate evaluation metrics and confusion matrix for your pitch deck
    evaluate_test_set(best_weights, f"{DATASET_DIR}/test")