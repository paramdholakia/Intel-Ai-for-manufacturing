import os
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './app/static/uploads'
app.config['MODEL_FOLDER'] = './models'

IMAGE_SIZE = (256, 256)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

models = {}

def load_all_models():
    models.clear()
    for model_file in os.listdir(app.config['MODEL_FOLDER']):
        if model_file.endswith('_best.h5'):
            category = model_file.replace('_best.h5', '')
            model_path = os.path.join(app.config['MODEL_FOLDER'], model_file)
            try:
                models[category] = load_model(model_path, compile=False)
                print(f"Successfully loaded model for category: {category}")
            except Exception as e:
                print(f"Error loading model {model_file}: {e}")

def preprocess_image(image_path, target_size=IMAGE_SIZE):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    return np.array(img) / 255.0

def generate_heatmap(original_image, anomaly_map, output_path, color_map='magma', alpha=0.6):
    heatmap_colored = plt.get_cmap(color_map)(anomaly_map)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    blended_image = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)
    Image.fromarray(blended_image).save(output_path)
    plt.close()

def generate_bw_heatmap(anomaly_map, output_path):
    threshold = np.mean(anomaly_map) + 2.5 * np.std(anomaly_map)
    threshold = np.clip(threshold, 0.2, 0.8)
    binary_map = (anomaly_map > threshold).astype(np.uint8) * 255
    cv2.imwrite(output_path, binary_map)

@app.route('/')
def index():
    return render_template('index.html', categories=models.keys(), results=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or request.files['file'].filename == '':
        return redirect(url_for('index'))

    file = request.files['file']
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    category = request.form.get('category')
    if category not in models:
        return redirect(url_for('index'))

    model = models[category]
    original_img_array = preprocess_image(filepath)
    original_img_display = (original_img_array * 255).astype(np.uint8)

    input_img = np.expand_dims(original_img_array, axis=0)
    reconstructed_img_array = model.predict(input_img)[0]

    mse = np.mean(np.square(original_img_array - reconstructed_img_array), axis=-1)
    anomaly_map_smoothed = cv2.GaussianBlur(mse, (5, 5), 0)
    vmax = np.percentile(anomaly_map_smoothed, 99)
    anomaly_map_clipped = np.clip(anomaly_map_smoothed, 0, vmax)
    anomaly_map_normalized = (anomaly_map_clipped - np.min(anomaly_map_clipped)) / (np.max(anomaly_map_clipped) - np.min(anomaly_map_clipped))

    # --- Generate and Save Images ---
    reconstructed_img_display = (reconstructed_img_array * 255).astype(np.uint8)
    reconstructed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f'reconstructed_{filename}')
    Image.fromarray(reconstructed_img_display).save(reconstructed_image_path)

    heatmap_color_path = os.path.join(app.config['UPLOAD_FOLDER'], f'heatmap_color_{filename}')
    generate_heatmap(original_img_display, anomaly_map_normalized, heatmap_color_path)

    heatmap_bw_path = os.path.join(app.config['UPLOAD_FOLDER'], f'heatmap_bw_{filename}')
    generate_bw_heatmap(anomaly_map_normalized, heatmap_bw_path)

    difference_image = (np.abs(original_img_array - reconstructed_img_array) * 255).astype(np.uint8)
    difference_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f'difference_{filename}')
    Image.fromarray(difference_image).save(difference_image_path)

    peak_error = np.max(mse)
    anomaly_percentage = (np.sum(anomaly_map_normalized > 0.5) / anomaly_map_normalized.size) * 100

    results = {
        'original_image': f'/static/uploads/{filename}',
        'reconstructed_image': f'/static/uploads/reconstructed_{filename}',
        'heatmap_color': f'/static/uploads/heatmap_color_{filename}',
        'heatmap_bw': f'/static/uploads/heatmap_bw_{filename}',
        'difference_image': f'/static/uploads/difference_{filename}',
        'peak_error': f'{peak_error:.4f}',
        'anomaly_percentage': f'{anomaly_percentage:.2f}'
    }

    return render_template('index.html', categories=models.keys(), results=results, selected_category=category)

load_all_models()

if __name__ == '__main__':
    app.run(debug=True)