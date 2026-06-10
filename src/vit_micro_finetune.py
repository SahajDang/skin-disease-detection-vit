import os
import pandas as pd
import tensorflow as tf
import keras

keras.config.enable_unsafe_deserialization()

from sklearn.model_selection import train_test_split
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


print("TensorFlow:", tf.__version__)

# ----------------------
# PATHS
# ----------------------
base_path = "/Users/sahajdang/Documents/dataset"
image_dir = os.path.join(base_path, "images")
csv_path = os.path.join(base_path, "classification_ready.csv")

IMG_SIZE = 224
BATCH_SIZE = 16

# ----------------------
# DATA
# ----------------------
df = pd.read_csv(csv_path)
df["image"] = df["image"] + ".jpg"

train_df, temp_df = train_test_split(
    df,
    test_size=0.3,
    stratify=df["label"],
    random_state=42
)

val_df, _ = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["label"],
    random_state=42
)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
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
# FREEZE EVERYTHING
# ----------------------
for layer in model.layers:
    layer.trainable = False

# ----------------------
# UNFREEZE LAST 3 TRANSFORMER BLOCKS
# ----------------------
for layer in model.layers:
    if "Transformer_encoderblock_9" in layer.name or \
       "Transformer_encoderblock_10" in layer.name or \
       "Transformer_encoderblock_11" in layer.name:
        layer.trainable = True

print("🔥 Last 3 transformer blocks unfrozen")

# ----------------------
# COMPILE WITH VERY SMALL LR
# ----------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-7),
    loss=FocalLoss(),
    metrics=["accuracy"]
)

# ----------------------
# TRAIN (SHORT)
# ----------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=3
)

# ----------------------
# SAVE NEW VERSION
# ----------------------
model.save("vit_micro_finetuned.keras")

print("\n✅ Micro Fine-Tuning Complete")