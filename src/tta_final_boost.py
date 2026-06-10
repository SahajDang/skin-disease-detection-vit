import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras

keras.config.enable_unsafe_deserialization()

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from vit_keras.layers import ClassToken, AddPositionEmbs, TransformerBlock

# ----------------------
# FOCAL LOSS (same as training)
# ----------------------
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25,
                 name="focal_loss",
                 reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE):
        super().__init__(name=name, reduction=reduction)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = self.alpha * tf.pow(1 - y_pred, self.gamma)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha,
        })
        return config

# ----------------------
# PATHS
# ----------------------
base_path = "/Users/sahajdang/Documents/dataset"
image_dir = os.path.join(base_path, "images")
csv_path = os.path.join(base_path, "classification_ready.csv")

IMG_SIZE = 224
BATCH_SIZE = 16
TTA_ROUNDS = 5

# ----------------------
# DATA SPLIT (same seed!)
# ----------------------
df = pd.read_csv(csv_path)
df["image"] = df["image"] + ".jpg"

train_df, temp_df = train_test_split(
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

# ----------------------
# LOAD MODEL
# ----------------------
model = tf.keras.models.load_model(
    "vit_micro_finetuned.keras",
    custom_objects={
        "FocalLoss": FocalLoss,
        "ClassToken": ClassToken,
        "AddPositionEmbs": AddPositionEmbs,
        "TransformerBlock": TransformerBlock
    },
    safe_mode=False
)

print("✅ Model Loaded")

# ----------------------
# TTA GENERATOR
# ----------------------
tta_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True,
    width_shift_range=0.05,
    height_shift_range=0.05
)

test_gen = tta_datagen.flow_from_dataframe(
    test_df,
    directory=image_dir,
    x_col="image",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ----------------------
# TTA INFERENCE
# ----------------------
print("\n🚀 Running TTA...")

predictions = []

for i in range(TTA_ROUNDS):
    print(f"TTA round {i+1}/{TTA_ROUNDS}")
    preds = model.predict(test_gen, verbose=1)
    predictions.append(preds)

final_preds = np.mean(predictions, axis=0)

y_true = test_gen.classes
y_pred = np.argmax(final_preds, axis=1)

accuracy = np.mean(y_true == y_pred)

print("\n🔥 FINAL TTA Accuracy:", accuracy)