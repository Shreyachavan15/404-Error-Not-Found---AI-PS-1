import os

dataset_path = "test"  # your 6-class folder
for class_name in os.listdir(dataset_path):
    class_folder = os.path.join(dataset_path, class_name)
    images = sorted(os.listdir(class_folder))
    for idx, img_name in enumerate(images):
        ext = os.path.splitext(img_name)[1]  # keep extension (.jpg)
        new_name = f"{class_name}_{idx+1}{ext}"
        os.rename(os.path.join(class_folder, img_name), os.path.join(class_folder, new_name))