import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras

keras.config.enable_unsafe_deserialization()

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from vit_keras.layers import ClassToken, AddPositionEmbs, TransformerBlock

# ----------------------
# FOCAL LOSS
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
# DATA
# ----------------------
base_path = "/Users/sahajdang/Documents/dataset"
image_dir = os.path.join(base_path, "images")
csv_path = os.path.join(base_path, "classification_ready.csv")

IMG_SIZE = 224
BATCH_SIZE = 16

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

datagen = ImageDataGenerator(rescale=1./255)

test_gen = datagen.flow_from_dataframe(
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
# LOAD MODEL
# ----------------------
model = tf.keras.models.load_model(
    "vit_focal_stable.keras",
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
# PREDICT
# ----------------------
preds = model.predict(test_gen)

y_true = test_gen.classes
y_pred = np.argmax(preds, axis=1)

print("\n📊 Classification Report:\n")
print(classification_report(y_true, y_pred))

print("\n📉 Confusion Matrix:\n")
print(confusion_matrix(y_true, y_pred))