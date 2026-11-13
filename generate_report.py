import numpy as np
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. UPDATED BASE FOLDER: training_model-plant
# 2. UPDATED DATASET: phytoguard_1k
DATASET_DIR = r"C:\Users\User\Documents\training_model-plant\phytoguard_1k"

# 3. VERIFY THIS FOLDER NAME (it might be -3 or -4 after your most recent 1K training)
BEST_WEIGHTS_PATH = r"C:\Users\User\Documents\training_model-plant\runs\phytoguard_16cls-3\weights\best.pt"

def evaluate_test_set():
    print(f"Loading weights from: {BEST_WEIGHTS_PATH}")
    model = YOLO(BEST_WEIGHTS_PATH)
    results = model.predict(source=f"{DATASET_DIR}/test/**/*", stream=True, verbose=False)

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
    
    with open(r"C:\Users\User\Documents\training_model-plant\phytoguard_eval_report.txt", "w") as f:
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
    plt.savefig(r"C:\Users\User\Documents\training_model-plant\phytoguard_confusion_matrix.png", dpi=300)
    print("[SAVED] Confusion Matrix -> phytoguard_confusion_matrix.png")

if __name__ == "__main__":
    evaluate_test_set()