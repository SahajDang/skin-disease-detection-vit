import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models
from vit_keras import vit  # IMPORTANT: must import before loading model

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

print("\nTrain samples:", len(train_df))

# ================================
# GENERATOR (NO AUGMENTATION)
# ================================
datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_dataframe(
    train_df,
    directory=image_dir,
    x_col="image",
    y_col="label",
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

print("\nClasses:", train_gen.class_indices)

# ================================
# LOAD TRAINED MODEL
# ================================
model = tf.keras.models.load_model(
    "vit_skin_model.keras",
    compile=False,
    safe_mode=False
)

print("\nModel loaded successfully.")

# ================================
# CREATE EMBEDDING MODEL
# ================================
# Extract the feature layer before classifier head
embedding_layer = model.get_layer("batch_normalization").output

embedding_model = models.Model(
    inputs=model.input,
    outputs=embedding_layer
)

print("Embedding model ready.")

# ================================
# EXTRACT EMBEDDINGS
# ================================
embeddings = embedding_model.predict(train_gen, verbose=1)
labels = train_gen.classes

print("Embeddings shape:", embeddings.shape)

# ================================
# SAVE
# ================================
np.save("train_embeddings.npy", embeddings)
np.save("train_labels.npy", labels)

print("\n✅ Train embeddings saved successfully.")