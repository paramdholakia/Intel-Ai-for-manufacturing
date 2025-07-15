import os
import cv2
import numpy as np
from PIL import Image

def get_product_categories(dataset_path):
    categories = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d)) and d != 'good']
    return categories

def preprocess_image(image_path, target_size=(256, 256)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    return img_array

def load_good_images(dataset_path, category, target_size=(256, 256)):
    good_images_path = os.path.join(dataset_path, category, 'train', 'good')
    images = []
    for img_name in os.listdir(good_images_path):
        img_path = os.path.join(good_images_path, img_name)
        if os.path.isfile(img_path):
            preprocessed_img = preprocess_image(img_path, target_size)
            images.append(preprocessed_img)
    return np.array(images)

if __name__ == '__main__':
    dataset_root = 'D:/Coding/Internship/Intel AI for mfg/Dataset'
    categories = get_product_categories(dataset_root)
    print(f"Found product categories: {categories}")

    # Example usage: Load good images for 'bottle' category
    if 'bottle' in categories:
        bottle_good_images = load_good_images(dataset_root, 'bottle')
        print(f"Loaded {len(bottle_good_images)} good images for 'bottle' category.")
        print(f"Image shape: {bottle_good_images.shape}")
