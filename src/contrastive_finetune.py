import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras

# 🔥 REQUIRED FOR VIT + LAMBDA
keras.config.enable_unsafe_deserialization()

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from vit_keras.layers import ClassToken, AddPositionEmbs, TransformerBlock

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

# ================================
# LOAD DATA
# ================================
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

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_dataframe(
    train_df,
    directory=image_dir,
    x_col="image",
    y_col="label",
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical"
)

val_gen = val_datagen.flow_from_dataframe(
    val_df,
    directory=image_dir,
    x_col="image",
    y_col="label",
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical"
)

num_classes = len(train_gen.class_indices)

# ================================
# LOAD EXISTING ViT MODEL
# ================================
model = tf.keras.models.load_model(
    "vit_skin_model.keras",
    custom_objects={
        "ClassToken": ClassToken,
        "AddPositionEmbs": AddPositionEmbs,
        "TransformerBlock": TransformerBlock
    },
    compile=False,
    safe_mode=False
)

print("\nModel loaded successfully.")

# ================================
# REMOVE CLASSIFIER HEAD
# ================================
embedding_layer = model.get_layer("batch_normalization").output

# Normalize embeddings for contrastive learning
embedding_output = layers.Lambda(
    lambda x: tf.math.l2_normalize(x, axis=1),
    name="normalized_embedding"
)(embedding_layer)

embedding_model = models.Model(
    inputs=model.input,
    outputs=embedding_output
)

print("Embedding model ready.")

# ================================
# ADD NEW CLASSIFIER WITH MARGIN HEAD
# ================================
x = layers.Dense(512, activation="relu")(embedding_output)
x = layers.Dropout(0.4)(x)
output = layers.Dense(num_classes, activation="softmax")(x)

final_model = models.Model(
    inputs=model.input,
    outputs=output
)

# ================================
# FINE-TUNE STRATEGY
# ================================
for layer in final_model.layers[:-30]:
    layer.trainable = False

final_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

final_model.summary()

# ================================
# TRAIN
# ================================
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True)
]

print("\n========== CONTRASTIVE FINE-TUNING ==========")

final_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5,
    callbacks=callbacks
)

# ================================
# SAVE MODEL
# ================================
final_model.save("vit_contrastive_improved.keras")

print("\nContrastive fine-tuning complete.")