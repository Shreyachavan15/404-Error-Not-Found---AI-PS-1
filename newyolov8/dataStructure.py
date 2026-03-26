import os
import shutil
import random

input_folder = "test"
label_folder = "test_labels"
output_folder = "test_yolo"
os.makedirs(output_folder, exist_ok=True)

for split in ["train", "valid"]:
    os.makedirs(os.path.join(output_folder, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, split, "labels"), exist_ok=True)

for class_name in os.listdir(input_folder):
    images = os.listdir(os.path.join(input_folder, class_name))
    random.shuffle(images)
    split_idx = int(len(images)*0.8)
    train_imgs = images[:split_idx]
    valid_imgs = images[split_idx:]

    for img_list, split in [(train_imgs,"train"), (valid_imgs,"valid")]:
        for img_name in img_list:
            # copy image
            shutil.copy(os.path.join(input_folder, class_name, img_name),
                        os.path.join(output_folder, split, "images", img_name))
            # copy label
            label_file = os.path.join(label_folder, class_name, os.path.splitext(img_name)[0] + ".txt")
            if os.path.exists(label_file):
                shutil.copy(label_file, os.path.join(output_folder, split, "labels", os.path.splitext(img_name)[0] + ".txt"))