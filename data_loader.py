# data_loader.py

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

def load_crack_data_by_folder(base_dir, img_size=(448, 448), threshold=0.5):
    images = []
    labels = []
    
    def sort_key(folder_name):
        try:
            return int(folder_name)
        except ValueError:
            return float('inf')

    label_folders = sorted(os.listdir(base_dir), key=sort_key)
    print(f"Label Folders: {label_folders}")

    for label_idx, label_folder in enumerate(label_folders):
        label_path = os.path.join(base_dir, label_folder)
        if not os.path.isdir(label_path):
            continue

        for filename in os.listdir(label_path):
            if filename.lower().endswith(('.jpg', '.png')):
                img_path = os.path.join(label_path, filename)
                try:
                    img = Image.open(img_path).convert("L").resize(img_size)
                    img = np.array(img) / 255.0
                    binary_img = (img > threshold).astype(np.float32)

                    images.append(binary_img)
                    labels.append(label_idx)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")

    return np.array(images), np.array(labels)

def load_and_split_data(base_dir, img_size=(448, 448), test_size=0.2, random_state=42):
    images, labels = load_crack_data_by_folder(base_dir, img_size=img_size)
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    return train_images, test_images, train_labels, test_labels
