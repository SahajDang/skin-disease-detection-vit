import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing import image
from vit_keras import visualize

# ================================
# SETTINGS
# ================================
img_size = 384
model_path = "vit_skin_model.keras"

# CHANGE THIS TO ANY IMAGE YOU WANT TO TEST
img_path = "/Users/sahajdang/Documents/dataset/images/ISIC_0024306.jpg"

# ================================
# LOAD MODEL (SAFE MODE FIX)
# ================================
model = tf.keras.models.load_model(
    model_path,
    compile=False,        # No need to compile for visualization
    safe_mode=False       # Needed because ViT uses Lambda layer
)

print("Model loaded successfully.")

# ================================
# LOAD & PREPROCESS IMAGE
# ================================
img = image.load_img(img_path, target_size=(img_size, img_size))
img_array = image.img_to_array(img)
img_array = img_array / 255.0

input_tensor = np.expand_dims(img_array, axis=0)

# ================================
# PREDICTION
# ================================
pred = model.predict(input_tensor)
predicted_class = np.argmax(pred)
confidence = np.max(pred)

print("Predicted Class:", predicted_class)
print("Confidence:", confidence)

# ================================
# ATTENTION MAP (FIXED INPUT)
# ================================
# vit_keras expects raw uint8 image (not normalized)
img_for_attention = cv2.imread(img_path)
img_for_attention = cv2.cvtColor(img_for_attention, cv2.COLOR_BGR2RGB)

attention_map = visualize.attention_map(model, img_for_attention)

# ================================
# DISPLAY RESULTS
# ================================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_array)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(attention_map)
plt.title("ViT Attention Map")
plt.axis("off")

plt.show()