import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from vit_keras import vit
from vit_keras.layers import ClassToken, AddPositionEmbs, TransformerBlock

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))

# =========================
# PATHS
# =========================
base_path = "/Users/sahajdang/Documents/dataset"
image_dir = os.path.join(base_path, "images")
csv_path = os.path.join(base_path, "classification_ready.csv")

IMG_SIZE = 224
BATCH_SIZE = 16

# =========================
# LOAD DATA
# =========================
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

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.15
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_dataframe(
    train_df,
    directory=image_dir,
    x_col="image",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_gen = val_datagen.flow_from_dataframe(
    val_df,
    directory=image_dir,
    x_col="image",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

num_classes = len(train_gen.class_indices)

# =========================
# FOCAL LOSS
# =========================
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = self.alpha * tf.pow(1 - y_pred, self.gamma)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=1)

# =========================
# BUILD VIT 224
# =========================
vit_model = vit.vit_b16(
    image_size=IMG_SIZE,
    activation='softmax',
    pretrained=True,
    include_top=False,
    pretrained_top=False,
    classes=num_classes
)

x = vit_model.output
x = layers.Dense(512, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs=vit_model.input, outputs=output)

print("✅ ViT built")

# =========================
# FREEZE BACKBONE FIRST
# =========================
for layer in vit_model.layers:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=FocalLoss(),
    metrics=["accuracy"]
)

print("\n========== PHASE 1: HEAD TRAINING ==========")

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5,
    callbacks=[
        EarlyStopping(patience=2, restore_best_weights=True)
    ]
)

# =========================
# UNFREEZE LAST 2 BLOCKS
# =========================
for layer in vit_model.layers[-20:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss=FocalLoss(),
    metrics=["accuracy"]
)

print("\n========== PHASE 2: PARTIAL FINE-TUNE ==========")

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=8,
    callbacks=[
        EarlyStopping(patience=3, restore_best_weights=True),
        ReduceLROnPlateau(patience=2, factor=0.3)
    ]
)

# =========================
# FINAL EVALUATION
# =========================
loss, acc = model.evaluate(val_gen)
print("\n🔥 FINAL Validation Accuracy:", acc)

model.save("vit_focal_stable.keras")