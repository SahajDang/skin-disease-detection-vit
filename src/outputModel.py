# =========================
# IMPORTS
# =========================
import keras
keras.config.enable_unsafe_deserialization()   # 🔥 required for your model

import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ViT custom layers
from vit_keras.layers import ClassToken, AddPositionEmbs

# =========================
# CONFIG
# =========================
MODEL_PATH = "vit_skin_model_hard_mined.keras"   # your model
IMG_SIZE = (384, 384)

# Folder containing test images (change if needed)
IMAGE_FOLDER = "dataset/test/MEL"

# Class labels (must match training)
class_names = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']

# =========================
# LOAD MODEL
# =========================
print("🔄 Loading model...")

model = load_model(
    MODEL_PATH,
    custom_objects={
        "ClassToken": ClassToken,
        "AddPositionEmbs": AddPositionEmbs
    },
    compile=False
)

print("✅ Model loaded successfully!")

# =========================
# FUNCTION: PREDICT IMAGE
# =========================
def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)[0]

    # Top prediction
    pred_index = np.argmax(preds)
    pred_class = class_names[pred_index]
    confidence = preds[pred_index]

    # Top-3 predictions
    top3_idx = np.argsort(preds)[-3:][::-1]
    top3 = [(class_names[i], preds[i]) for i in top3_idx]

    return img, pred_class, confidence, top3

# =========================
# PROCESS MULTIPLE IMAGES
# =========================
print("🔄 Running predictions on sample images...\n")

image_files = os.listdir(IMAGE_FOLDER)

# Take first 3 images for demo
image_files = image_files[:3]

for idx, img_file in enumerate(image_files):

    img_path = os.path.join(IMAGE_FOLDER, img_file)

    img, pred_class, confidence, top3 = predict_image(img_path)

    print(f"📸 Image: {img_file}")
    print(f"👉 Prediction: {pred_class} ({confidence:.4f})")

    print("Top-3 Predictions:")
    for cls, conf in top3:
        print(f"   {cls}: {conf:.4f}")
    print("-" * 40)

    # =========================
    # DISPLAY IMAGE
    # =========================
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title(f"{pred_class} ({confidence:.2f})")
    plt.axis("off")

    # Save output image
    save_name = f"prediction_{idx+1}.png"
    plt.savefig(save_name, dpi=300)
    plt.close()

    print(f"✅ Saved: {save_name}\n")

print("🎉 All predictions completed!")