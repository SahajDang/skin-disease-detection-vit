import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from vit_keras import vit

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))

# ================================
# PATHS
# ================================
base_path = "/Users/sahajdang/Documents/dataset"
image_dir = os.path.join(base_path, "images")
csv_path = os.path.join(base_path, "classification_ready.csv")

img_size = 384
batch_size = 8
num_classes = 7

# ================================
# LOAD DATA
# ================================
df = pd.read_csv(csv_path)
df["image"] = df["image"] + ".jpg"

_, temp_df = train_test_split(
    df,
    test_size=0.3,
    stratify=df["label"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["label"],
    random_state=42
)

datagen = ImageDataGenerator(rescale=1./255)

test_gen = datagen.flow_from_dataframe(
    test_df,
    directory=image_dir,
    x_col="image",
    y_col="label",
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

# ================================
# BUILD SAME ARCHITECTURE
# ================================

# Base ViT WITHOUT top
vit_backbone = vit.vit_b16(
    image_size=img_size,
    pretrained=False,
    include_top=False,
    pretrained_top=False
)

x = vit_backbone.output  # (None, 768)

# Your custom head
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs=vit_backbone.input, outputs=outputs)

# Load weights
model.load_weights("vit_contrastive_improved.keras")

print("✅ Correct weights loaded successfully.")

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ================================
# TTA
# ================================
def tta_predict(model, generator, rounds=3):
    predictions = []

    for i in range(rounds):
        print(f"TTA round {i+1}/{rounds}")
        preds = model.predict(generator, verbose=1)
        predictions.append(preds)

    return np.mean(predictions, axis=0)

print("\n🚀 Running TTA...\n")

tta_preds = tta_predict(model, test_gen, rounds=3)

y_true = test_gen.classes
y_pred = np.argmax(tta_preds, axis=1)

accuracy = np.mean(y_pred == y_true)

print("\n🔥 FINAL TTA Accuracy:", accuracy)