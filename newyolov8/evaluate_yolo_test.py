import os
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------
# PATHS
# -----------------------
test_images_folder = "test/images"
test_labels_folder = "test/labels"

# -----------------------
# CLASS NAMES
# -----------------------
class_names = ['PET','PP','PS','HDPE','LDPE','Others']
num_classes = len(class_names)

# -----------------------
# LOAD MODEL
# -----------------------
model_path = "runs/detect/plastic_detector_auto11/weights/best.pt"

print("🚀 Loading YOLO model...")
model = YOLO(model_path)

# -----------------------
# EVALUATION
# -----------------------
y_true = []
y_pred = []

print("🔍 Running evaluation...")

for img_file in os.listdir(test_images_folder):

    if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    base_name = os.path.splitext(img_file)[0]

    img_path = os.path.join(test_images_folder, img_file)
    label_path = os.path.join(test_labels_folder, base_name + ".txt")

    # ❌ Skip if no label
    if not os.path.exists(label_path):
        print(f"⚠ Missing label for {img_file}")
        continue

    # -----------------------
    # READ GROUND TRUTH
    # -----------------------
    with open(label_path, "r") as f:
        lines = f.readlines()

    if len(lines) == 0:
        continue

    true_classes = [int(line.split()[0]) for line in lines]

    # -----------------------
    # PREDICTION
    # -----------------------
    results = model.predict(img_path, conf=0.25, verbose=False)

    if len(results[0].boxes) == 0:
        pred_classes = [num_classes - 1] * len(true_classes)  # Others
    else:
        pred_classes = [int(box.cls) for box in results[0].boxes]

    # -----------------------
    # MATCH LENGTH
    # -----------------------
    min_len = min(len(true_classes), len(pred_classes))

    if min_len == 0:
        continue

    y_true.extend(true_classes[:min_len])
    y_pred.extend(pred_classes[:min_len])

# -----------------------
# RESULTS
# -----------------------
print("\n📊 Total samples:", len(y_true))

if len(y_true) == 0:
    print("❌ ERROR: No valid data found!")
    exit()

labels = list(range(num_classes))

print("\n📊 Classification Report:\n")
print(classification_report(
    y_true,
    y_pred,
    labels=labels,
    target_names=class_names,
    zero_division=0
))

print("\n📊 Confusion Matrix:\n")
cm = confusion_matrix(y_true, y_pred, labels=labels)
print(cm)

print("\n✅ EVALUATION COMPLETE!")

