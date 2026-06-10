import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
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

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["label"],
    random_state=42
)

print("Original Train Size:", len(train_df))

# IMPORTANT:
# Reset index to match generator order
train_df = train_df.reset_index(drop=True)

# ================================
# LOAD HARD INDICES
# ================================
hard_indices = np.load("hard_nv_indices.npy")
print("Hard NV samples:", len(hard_indices))

hard_samples_df = train_df.iloc[hard_indices]

# Duplicate hard samples (2x emphasis)
train_df_emphasized = pd.concat([
    train_df,
    hard_samples_df,
    hard_samples_df
]).reset_index(drop=True)

print("New Train Size:", len(train_df_emphasized))

# ================================
# GENERATORS
# ================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_dataframe(
    train_df_emphasized,
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

# ================================
# LOAD EXISTING MODEL
# ================================
model = tf.keras.models.load_model(
    "vit_skin_model.keras",
    safe_mode=False,
    compile=False
)

print("Model loaded successfully.")

# Light fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-6),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ================================
# RETRAIN (SHORT + SAFE)
# ================================
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5,
    callbacks=[EarlyStopping(patience=2, restore_best_weights=True)]
)

model.save("vit_skin_model_hard_mined.keras")

print("Hard mining retraining complete.")