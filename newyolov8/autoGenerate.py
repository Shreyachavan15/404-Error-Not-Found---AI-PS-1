import cv2
import os

# Map folder names to class IDs
class_map = {
    "HDPE": 0,
    "LDPE": 1,
    "Other": 2,
    "PET": 3,
    "PP": 4,
    "PS": 5
}

dataset_path = "test/images"
label_output_path = "test_labels"
os.makedirs(label_output_path, exist_ok=True)

for class_name, class_id in class_map.items():
    class_folder = os.path.join(dataset_path, class_name)
    label_class_folder = os.path.join(label_output_path, class_name)
    os.makedirs(label_class_folder, exist_ok=True)

    for img_name in os.listdir(class_folder):
        img_path = os.path.join(class_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        height, width, _ = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        label_lines = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            x_center = (x + w/2) / width
            y_center = (y + h/2) / height
            w_norm = w / width
            h_norm = h / height
            label_lines.append(f"{class_id} {x_center} {y_center} {w_norm} {h_norm}")

        label_file = os.path.join(label_class_folder, os.path.splitext(img_name)[0] + ".txt")
        with open(label_file, "w") as f:
            f.write("\n".join(label_lines))