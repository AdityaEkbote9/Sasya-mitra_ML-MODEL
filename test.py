import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path

# --- 1. CONFIGURATION ---

TEST_IMAGE_PATH = r"C:\Users\User\Downloads\OIP.webp"
# Point this to your actual trained weights (-2 folder based on your previous logs)
WEIGHTS_PATH = r"C:\Users\User\Documents\training-plant\runs\phytoguard_16cls-3\weights\best.pt"

def calculate_entropy(probs):
    """Calculate Shannon Entropy to measure model uncertainty (FIELDPROOF Engine)"""
    probs = np.clip(probs, 1e-7, 1.0) # Prevent log(0)
    entropy = -np.sum(probs * np.log(probs))
    return entropy

def test_image(image_path, weights_path):
    print(f"\n[INFO] Loading FIELDPROOF Engine with {weights_path}...")
    model = YOLO(weights_path)
    
    if not Path(image_path).exists():
        print(f"[ERROR] Image not found at {image_path}")
        return

    # Run inference
    results = model.predict(source=image_path, imgsz=224, verbose=False)
    
    # Extract prediction data
    result = results[0]
    probs = result.probs.data.cpu().numpy()
    top1_idx = result.probs.top1
    top1_conf = result.probs.top1conf.item()
    predicted_class = model.names[top1_idx]
    
    # Calculate FIELDPROOF Uncertainty
    entropy = calculate_entropy(probs)
    
    # FIELDPROOF Routing Logic
    if top1_conf < 0.75 or entropy > 1.15:
        status = "⚠️ UNCERTAIN: ROUTE TO KVK EXPERT"
        color = "red"
    else:
        status = "✅ HIGH CONFIDENCE: AUTO-VERIFIED"
        color = "green"

    # Print Terminal Report
    print("\n" + "="*50)
    print(" 🔬 PHYTOGUARD INFERENCE REPORT")
    print("="*50)
    print(f"File:       {Path(image_path).name}")
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {top1_conf * 100:.2f}%")
    print(f"Entropy:    {entropy:.4f} (Threshold: 1.15)")
    print(f"Status:     {status}")
    print("="*50 + "\n")

    # Visual Overlay (Simulating Dashboard Output)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis("off")
    
    # Format display text
    display_text = (
        f"Prediction: {predicted_class.replace('___', ' ')}\n"
        f"Confidence: {top1_conf * 100:.1f}%\n"
        f"Entropy: {entropy:.2f}\n"
        f"{status}"
    )
    
    # Add text overlay box
    plt.text(
        10, 10, display_text, 
        color='white', 
        fontsize=12, 
        fontweight='bold',
        bbox=dict(facecolor=color, alpha=0.8, edgecolor='none', boxstyle='round,pad=0.5'),
        va='top'
    )
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Ensure you replace TEST_IMAGE_PATH at the top with an actual file path before running!
    test_image(TEST_IMAGE_PATH, WEIGHTS_PATH)