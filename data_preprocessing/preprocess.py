# -*- coding: utf-8 -*-
"""
preprocess.py

This script provides utility functions for handling the image data required for training.
It includes functions to discover product categories from the dataset directory structure,
preprocess individual images, and load all 'good' quality images for a specific category.

This script can also be run standalone to test the data loading process.
"""

# ─── IMPORTS ─────────────────────────────────────────────────────────────────────
import os
import cv2
import numpy as np
from PIL import Image

# ─── DATA UTILITY FUNCTIONS ──────────────────────────────────────────────────────

def get_product_categories(dataset_path):
    """
    Scans the dataset directory to find all product category subdirectories.

    Args:
        dataset_path (str): The root path of the dataset.

    Returns:
        list: A list of strings, where each string is a category name.
    """
    # A category is considered any subdirectory that is not named 'good'.
    categories = [
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d)) and d != 'good'
    ]
    return categories

def preprocess_image(image_path, target_size=(256, 256)):
    """
    Loads an image from a file path, converts it to RGB, resizes it, and
    normalizes its pixel values to the [0, 1] range.

    Args:
        image_path (str): The full path to the image file.
        target_size (tuple): The desired output size (height, width).

    Returns:
        np.ndarray: The preprocessed image as a NumPy array.
    """
    # Open the image using Pillow and convert to RGB to handle different image modes.
    img = Image.open(image_path).convert('RGB')
    # Resize the image to the specified target size.
    img = img.resize(target_size)
    # Convert the PIL Image object to a NumPy array.
    img_array = np.array(img)
    # Normalize the pixel values from [0, 255] to [0.0, 1.0] for the model.
    return img_array / 255.0

def load_good_images(dataset_path, category, target_size=(256, 256)):
    """
    Loads all images from the 'train/good' subdirectory for a given category.

    Args:
        dataset_path (str): The root path of the dataset.
        category (str): The product category to load images from.
        target_size (tuple): The target size for preprocessing the images.

    Returns:
        np.ndarray: A NumPy array containing all the preprocessed 'good' images.
    """
    # Construct the path to the directory containing good training images.
    good_images_path = os.path.join(dataset_path, category, 'train', 'good')
    images = []
    print(f"Loading good images from: {good_images_path}")

    # Iterate over all files in the directory.
    for img_name in os.listdir(good_images_path):
        img_path = os.path.join(good_images_path, img_name)
        # Ensure the path points to a file.
        if os.path.isfile(img_path):
            # Preprocess the image and add it to the list.
            preprocessed_img = preprocess_image(img_path, target_size)
            images.append(preprocessed_img)

    # Convert the list of images into a single NumPy array.
    return np.array(images)

# ─── SCRIPT ENTRYPOINT ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    """
    This block is executed when the script is run directly.
    It serves as a test to demonstrate the functionality of the utility functions.
    """
    # Define the root directory of the dataset.
    # Note: This path may need to be adjusted depending on the system.
    dataset_root = 'D:/Coding/Internship/Intel AI for mfg/Dataset'

    print("--- Testing Data Preprocessing Utilities ---")
    # Get and print all available product categories.
    categories = get_product_categories(dataset_root)
    print(f"Found product categories: {categories}")

    # Example usage: Load all 'good' images for the 'bottle' category.
    if 'bottle' in categories:
        print("\n--- Loading example data for 'bottle' category ---")
        bottle_good_images = load_good_images(dataset_root, 'bottle')
        if bottle_good_images.size > 0:
            print(f"Successfully loaded {len(bottle_good_images)} good images for 'bottle' category.")
            print(f"Dataset shape: {bottle_good_images.shape}") # (num_images, height, width, channels)
        else:
            print("No images were loaded for the 'bottle' category.")
    print("--------------------------------------------")
