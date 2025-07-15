

import os
import sys
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

# ─── PROJECT ROOT AND GPU SETUP ──────────────────────────────────────────────────
# Add project root to system path for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure GPU for optimal performance
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Configured {len(gpus)} GPU(s) with memory growth.")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

# ─── CORE HYPERPARAMETERS ────────────────────────────────────────────────────────
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 250  # Increased epochs with early stopping to ensure convergence
VAL_SPLIT = 0.15 # Using a slightly larger validation set for more reliable metrics
LEARNING_RATE = 0.001
PATIENCE_ES = 15  # More patience for early stopping
PATIENCE_LR = 7   # More patience for learning rate reduction
AUTOTUNE = tf.data.AUTOTUNE

# ─── ADVANCED LOSS FUNCTION (MSE + SSIM) ─────────────────────────────────────────
def combined_loss(y_true, y_pred):
    """
    A combination of Mean Squared Error (MSE) and Structural Similarity Index (SSIM)
    to capture both pixel-level accuracy and perceptual, structural details.
    """
    # Ensure y_true and y_pred are in the expected float32 format
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    ssim_loss = 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    
    # Weighting: 85% MSE, 15% SSIM. A common balance for image reconstruction.
    return 0.85 * mse_loss + 0.15 * ssim_loss

# ─── HIGH-PERFORMANCE TF.DATA PIPELINE ───────────────────────────────────────────
def create_dataset(filepaths, is_training=True):
    """
    Builds a tf.data pipeline for efficient loading and augmentation.
    """
    def parse_image(path):
        image_bytes = tf.io.read_file(path)
        image = tf.image.decode_png(image_bytes, channels=3)
        image = tf.image.resize(image, IMAGE_SIZE)
        return tf.cast(image, tf.float32) / 255.0

    def augment(image):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        # Use a smaller delta for brightness to avoid drastic changes
        image = tf.image.random_brightness(image, max_delta=0.05)
        return image

    dataset = tf.data.Dataset.from_tensor_slices(filepaths)
    dataset = dataset.map(parse_image, num_parallel_calls=AUTOTUNE)
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(filepaths))
        dataset = dataset.map(augment, num_parallel_calls=AUTOTUNE)

    # The model expects a tuple of (input, target), for an autoencoder, they are the same
    dataset = dataset.map(lambda x: (x, x), num_parallel_calls=AUTOTUNE)
    
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset

# ─── U-NET STYLE AUTOENCODER FOR PRECISION ───────────────────────────────────────
def build_unet_autoencoder(input_shape):
    """
    Constructs a U-Net-like autoencoder. Skip connections help preserve spatial
    information, leading to more precise reconstructions, which is critical for
    detecting subtle anomalies.
    """
    inputs = layers.Input(shape=input_shape)

    # Encoder Path
    s1 = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    p1 = layers.MaxPooling2D(2, padding='same')(s1)
    
    s2 = layers.Conv2D(64, 3, padding='same', activation='relu')(p1)
    p2 = layers.MaxPooling2D(2, padding='same')(s2)
    
    s3 = layers.Conv2D(128, 3, padding='same', activation='relu')(p2)
    p3 = layers.MaxPooling2D(2, padding='same')(s3)

    # Bottleneck
    b = layers.Conv2D(256, 3, padding='same', activation='relu')(p3)

    # Decoder Path (with skip connections)
    u3 = layers.UpSampling2D(2)(b)
    c3 = layers.Concatenate()([u3, s3])
    d3 = layers.Conv2D(128, 3, padding='same', activation='relu')(c3)

    u2 = layers.UpSampling2D(2)(d3)
    c2 = layers.Concatenate()([u2, s2])
    d2 = layers.Conv2D(64, 3, padding='same', activation='relu')(c2)

    u1 = layers.UpSampling2D(2)(d2)
    c1 = layers.Concatenate()([u1, s1])
    d1 = layers.Conv2D(32, 3, padding='same', activation='relu')(c1)

    # Output layer
    outputs = layers.Conv2D(input_shape[-1], 1, activation='sigmoid', padding='same')(d1)

    model = models.Model(inputs, outputs, name='unet_autoencoder')
    return model

# ─── MAIN TRAINING ORCHESTRATOR ──────────────────────────────────────────────────
# ─── MAIN TRAINING ORCHESTRATOR ──────────────────────────────────────────────────
def train_for_category(category, dataset_root, models_dir):
    print(f"\n{'='*60}\nTRAINING MODEL FOR CATEGORY: {category.upper()}\n{'='*60}")

    model_path = os.path.join(models_dir, f"{category}_best.h5")
    if os.path.exists(model_path):
        print(f"Model for {category} already exists at {model_path}. Skipping training.")
        return

    good_images_dir = os.path.join(dataset_root, category, 'train', 'good')
    if not os.path.isdir(good_images_dir):
        print(f"ERROR: Directory not found: {good_images_dir}. Skipping category.")
        return

    filepaths = [os.path.join(good_images_dir, f) for f in os.listdir(good_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not filepaths:
        print(f"ERROR: No images found in {good_images_dir}. Skipping category.")
        return

    # Split data into training and validation sets
    train_paths, val_paths = train_test_split(filepaths, test_size=VAL_SPLIT, random_state=42)
    print(f"Found {len(filepaths)} images. Using {len(train_paths)} for training and {len(val_paths)} for validation.")

    # Create tf.data pipelines
    train_ds = create_dataset(train_paths, is_training=True)
    val_ds = create_dataset(val_paths, is_training=False)

    # Build and compile the model
    model = build_unet_autoencoder((*IMAGE_SIZE, 3))
    model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE), loss=combined_loss)
    model.summary()

    # Define paths for saving model and logs
    cae_model_path = os.path.join(models_dir, f"{category}_cae.h5") # For compatibility with web app
    log_dir = os.path.join(models_dir, 'logs', category)
    os.makedirs(log_dir, exist_ok=True)

    # Define callbacks for robust training
    training_callbacks = [
        callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE_ES, verbose=1, restore_best_weights=True, min_delta=1e-5),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=PATIENCE_LR, min_lr=1e-6, verbose=1),
        callbacks.ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True, verbose=1, min_delta=1e-5),
        callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    ]

    # Start training
    print(f"\nStarting training for {category}...")
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=training_callbacks,
        verbose=1
    )

    print(f"\nTraining finished for {category}.")
    print(f"Best model saved at: {model_path}")
    
    # Save a copy with the _cae.h5 suffix for the Flask app
    model.save(cae_model_path)
    print(f"Saved compatibility model for web app at: {cae_model_path}")


if __name__ == '__main__':
    # Define root directories
    dataset_root = os.path.join(project_root, 'Dataset')
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Specify the categories to train. Add more categories here as needed.
    categories_to_train = ['bottle', 'cable', 'capsule']
    
    print(f"Dataset Root: {dataset_root}")
    print(f"Models Directory: {models_dir}")
    print(f"Categories to Train: {categories_to_train}")

    # Loop through each category and train a dedicated model
    for category in categories_to_train:
        train_for_category(category, dataset_root, models_dir)

    print("\nAll specified categories have been trained.")
