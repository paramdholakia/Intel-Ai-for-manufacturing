# Anomaly Detector

## Project Overview
The Anomaly Detector is a cutting-edge computer vision project designed to identify defects and anomalies in manufactured products. Leveraging deep learning techniques, specifically autoencoders, this application provides a robust solution for quality control in industrial settings. It's built to be intuitive, providing clear visual feedback on detected anomalies through heatmaps and binary anomaly maps.

# Note :
* **THIS PROJECT IS STILL UNDER DEVELOPMENT!!** If you don't see different products mentioned in MVTec Anomaly Detection Dataset, then it's in development and training and will be updated soon. thank you for your paitence.

### Present 
Bottle model

### Models to be added soon
* cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper



## How It Works
At its core, the Anomaly Detector utilizes a U-Net style autoencoder, a type of neural network trained to reconstruct input data.

1.  **Training Phase:** The autoencoder is exclusively trained on "good" (defect-free) images of products from the MVTec Anomaly Detection Dataset. During this phase, the model learns the intricate patterns and features that define a normal, healthy product.
2.  **Detection Phase:** When a new product image is fed into the trained autoencoder, the model attempts to reconstruct it.
3.  **Anomaly Mapping:** If the product image contains an anomaly, the autoencoder will struggle to reconstruct the anomalous region accurately because it has never "seen" such patterns during training. The difference between the original image and its reconstruction (the "reconstruction error") highlights these anomalous areas.
4.  **Visual Feedback:** This reconstruction error is then processed and visualized in two ways:
    *   **Heatmap:** A color-coded map overlaid on the original image, where warmer colors indicate higher reconstruction error (i.e., more significant anomalies).
    *   **Binary Anomaly Map:** A black-and-white image that clearly outlines the anomalous regions, making them stand out for quick inspection.

## Technology Stack
*   **Backend:** Flask (Python web framework)
*   **Deep Learning:** TensorFlow 2.x, Keras
*   **Image Processing:** OpenCV (cv2), NumPy, PIL (Pillow), Matplotlib, scikit-image
*   **Frontend:** HTML5, CSS3 (with a custom "Factory-Tech" theme), JavaScript (for interactive elements)
*   **Icons:** Font Awesome

## Dataset
This project utilizes the **MVTec Anomaly Detection Dataset**. This dataset is specifically designed for benchmarking anomaly detection algorithms and contains a wide variety of industrial products with different types of defects. For training, only the "good" images are used, allowing the autoencoder to learn the normal distribution of product features.

## Key Features
*   **Intuitive Single-Page Application (SPA):** Seamless user experience with image upload and results displayed on the same page.
*   **Factory-Themed UI:** A professional and engaging user interface designed to resemble an industrial control panel.
*   **Detailed Anomaly Analysis:** Provides original, reconstructed, difference, heatmap, and binary anomaly maps for comprehensive inspection.
*   **Quantitative Metrics:** Displays peak reconstruction error and anomaly percentage for objective assessment.
*   **Extensible Training System:** The backend training script is designed to easily accommodate new product categories.



## Setup and Local Usage Guide

### Prerequisites
*   Python 3.8+
*   Git

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/anomaly-detector.git
cd anomaly-detector
```
*(Note: Replace `https://github.com/your-username/anomaly-detector.git` with the actual repository URL if different.)*

### 2. Install Dependencies
It's highly recommended to use a virtual environment.
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Prepare the Dataset
Ensure your MVTec Anomaly Detection Dataset is structured as follows:
```
Intel_AI_for_mfg/
├── Dataset/
│   ├── bottle/
│   │   └── train/
│   │       └── good/  (contains good bottle images)
│   ├── cable/
│   │   └── train/
│   │       └── good/  (contains good cable images)
│   └── capsule/
│       └── train/
│           └── good/  (contains good capsule images)
└── ...
```
Place the `Dataset` folder in the root of the project directory.

### 4. Train the Models
The project is configured to train models for 'bottle', 'cable', and 'capsule' by default. You can extend this to other categories by modifying `training/train.py`.
```bash
python training/train.py
```
This script will save the trained models (`.h5` files) in the `models/` directory.

### 5. Run the Flask Application
```bash
python app/main.py
```
The application will typically run on `http://127.0.0.1:5000/`.

### 6. Access the Web Interface
Open your web browser and navigate to `http://127.0.0.1:5000/`.
You can now upload an image, select the product category, and click "Analyze" to see the anomaly detection results.

## Attribution
This project was developed by **Param Ashish Dholakia** for the **Intel AI for Manufacturing** internship program.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
