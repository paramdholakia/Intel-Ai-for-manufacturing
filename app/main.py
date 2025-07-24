# -*- coding: utf-8 -*-
"""
main.py

This script serves as the backend for the Anomaly Detector web application using Flask.
It handles file uploads, model loading, image preprocessing, anomaly prediction,
and visualization of the results.

Key functionalities include:
- A Flask web server to handle HTTP requests.
- Loading all pre-trained autoencoder models at startup.
- A main route to display the user interface.
- A prediction route that accepts an image and a product category, then:
    - Preprocesses the input image.
    - Uses the corresponding model to reconstruct the image.
    - Calculates the reconstruction error to generate an anomaly map.
    - Creates and saves several visualization images (reconstruction, heatmaps, etc.).
    - Renders the results back to the user on the web page.
"""

# ─── IMPORTS ─────────────────────────────────────────────────────────────────────
import os
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model # type: ignore
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# ─── FLASK APP INITIALIZATION AND CONFIGURATION ──────────────────────────────────
app = Flask(__name__)

# Configure paths for storing user-uploaded files and trained models.
app.config['UPLOAD_FOLDER'] = './app/static/uploads'
app.config['MODEL_FOLDER'] = './models'

# Define the standard image size for model input.
IMAGE_SIZE = (256, 256)

# Ensure the folder for uploaded images exists.
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# A global dictionary to hold the loaded models in memory.
models = {}

# ─── MODEL AND IMAGE HANDLING FUNCTIONS ──────────────────────────────────────────
def load_all_models():
    """
    Scans the model directory, loads all Keras models ending in '_best.h5',
    and stores them in the global `models` dictionary. This is done at startup
    to avoid loading models on every prediction request.
    """
    print("--- Loading all trained models ---")
    models.clear()
    for model_file in os.listdir(app.config['MODEL_FOLDER']):
        if model_file.endswith('_best.h5'):
            # Extract the category name from the filename.
            category = model_file.replace('_best.h5', '')
            model_path = os.path.join(app.config['MODEL_FOLDER'], model_file)
            try:
                # Load the model without compiling it (not needed for inference).
                models[category] = load_model(model_path, compile=False)
                print(f"Successfully loaded model for category: {category}")
            except Exception as e:
                print(f"Error loading model {model_file}: {e}")
    print("------------------------------------")

def preprocess_image(image_path, target_size=IMAGE_SIZE):
    """
    Loads, resizes, and normalizes an image from a given file path.

    Args:
        image_path (str): The path to the image file.
        target_size (tuple): The target size (height, width) for the image.

    Returns:
        np.ndarray: The preprocessed image as a NumPy array with values in [0, 1].
    """
    # Open the image and ensure it is in RGB format.
    img = Image.open(image_path).convert('RGB')
    # Resize the image to the required dimensions.
    img = img.resize(target_size)
    # Convert the image to a NumPy array and normalize pixel values to [0, 1].
    return np.array(img) / 255.0

def generate_heatmap(original_image, anomaly_map, output_path, color_map='magma', alpha=0.6):
    """
    Creates a color heatmap from an anomaly map and blends it with the original image.

    Args:
        original_image (np.ndarray): The original image (as a NumPy array).
        anomaly_map (np.ndarray): The grayscale anomaly map (values in [0, 1]).
        output_path (str): The path to save the resulting blended image.
        color_map (str): The matplotlib colormap to use for the heatmap.
        alpha (float): The transparency factor for blending the heatmap.
    """
    # Get the specified colormap from matplotlib.
    heatmap_colored = plt.get_cmap(color_map)(anomaly_map)[:, :, :3]
    # Convert the heatmap to an 8-bit unsigned integer format for image processing.
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    # Blend the original image and the heatmap.
    blended_image = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)
    # Save the resulting image.
    Image.fromarray(blended_image).save(output_path)
    # Close the matplotlib plot to free up memory.
    plt.close()

def generate_bw_heatmap(anomaly_map, output_path):
    """
    Creates a binary (black and white) anomaly map by thresholding the anomaly scores.

    Args:
        anomaly_map (np.ndarray): The grayscale anomaly map.
        output_path (str): The path to save the binary map image.
    """
    # Calculate a dynamic threshold to separate anomalous regions.
    threshold = np.mean(anomaly_map) + 2.5 * np.std(anomaly_map)
    # Clip the threshold to a reasonable range to avoid extreme values.
    threshold = np.clip(threshold, 0.2, 0.8)
    # Create a binary map where pixels above the threshold are white (255).
    binary_map = (anomaly_map > threshold).astype(np.uint8) * 255
    # Save the binary image.
    cv2.imwrite(output_path, binary_map)

# ─── FLASK ROUTES ────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    """
    Renders the main page of the web application.
    Passes the list of available model categories to the template.
    """
    return render_template('index.html', categories=models.keys(), results=None)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the prediction request. This is the core logic of the web application.
    It processes the uploaded file, performs anomaly detection, and returns the results.
    """
    # Basic validation: ensure a file was uploaded.
    if 'file' not in request.files or request.files['file'].filename == '':
        return redirect(url_for('index'))

    # Get the file and category from the request.
    file = request.files['file']
    filename = file.filename

    # Save the uploaded file to the server.
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Validate that a model exists for the selected category.
    category = request.form.get('category')
    if category not in models:
        return redirect(url_for('index'))

    # --- Anomaly Detection Process ---
    # 1. Load and preprocess the uploaded image.
    model = models[category]
    original_img_array = preprocess_image(filepath)
    original_img_display = (original_img_array * 255).astype(np.uint8)

    # 2. Get the model's reconstruction of the image.
    # The image needs an extra dimension to be processed as a batch of size 1.
    input_img = np.expand_dims(original_img_array, axis=0)
    reconstructed_img_array = model.predict(input_img)[0]

    # 3. Calculate the anomaly map.
    # The error is the mean squared difference between original and reconstruction.
    mse = np.mean(np.square(original_img_array - reconstructed_img_array), axis=-1)
    # Smooth the error map to reduce noise.
    anomaly_map_smoothed = cv2.GaussianBlur(mse, (5, 5), 0)
    # Clip extreme values for better visualization.
    vmax = np.percentile(anomaly_map_smoothed, 99)
    anomaly_map_clipped = np.clip(anomaly_map_smoothed, 0, vmax)
    # Normalize the map to a [0, 1] range.
    anomaly_map_normalized = (anomaly_map_clipped - np.min(anomaly_map_clipped)) / (np.max(anomaly_map_clipped) - np.min(anomaly_map_clipped))

    # --- Generate and Save Visualization Images ---
    # Reconstructed Image
    reconstructed_img_display = (reconstructed_img_array * 255).astype(np.uint8)
    reconstructed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f'reconstructed_{filename}')
    Image.fromarray(reconstructed_img_display).save(reconstructed_image_path)

    # Color Heatmap
    heatmap_color_path = os.path.join(app.config['UPLOAD_FOLDER'], f'heatmap_color_{filename}')
    generate_heatmap(original_img_display, anomaly_map_normalized, heatmap_color_path)

    # Black & White Anomaly Map
    heatmap_bw_path = os.path.join(app.config['UPLOAD_FOLDER'], f'heatmap_bw_{filename}')
    generate_bw_heatmap(anomaly_map_normalized, heatmap_bw_path)

    # Difference Image
    difference_image = (np.abs(original_img_array - reconstructed_img_array) * 255).astype(np.uint8)
    difference_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f'difference_{filename}')
    Image.fromarray(difference_image).save(difference_image_path)

    # --- Calculate Metrics ---
    peak_error = np.max(mse)
    anomaly_percentage = (np.sum(anomaly_map_normalized > 0.5) / anomaly_map_normalized.size) * 100

    # --- Prepare Results for Template ---
    # This dictionary will be passed to the HTML template to display the results.
    results = {
        'original_image': f'/static/uploads/{filename}',
        'reconstructed_image': f'/static/uploads/reconstructed_{filename}',
        'heatmap_color': f'/static/uploads/heatmap_color_{filename}',
        'heatmap_bw': f'/static/uploads/heatmap_bw_{filename}',
        'difference_image': f'/static/uploads/difference_{filename}',
        'peak_error': f'{peak_error:.4f}',
        'anomaly_percentage': f'{anomaly_percentage:.2f}'
    }

    # Render the main page again, but this time with the results.
    return render_template('index.html', categories=models.keys(), results=results, selected_category=category)

# ─── SCRIPT ENTRYPOINT ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    """
    This block is executed when the script is run directly.
    It loads all models and then starts the Flask development server.
    """
    load_all_models() # Load models into memory before starting the server.
    app.run(debug=True) # Run the app in debug mode for development.
