import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
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

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1
)

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
# REBUILD VIT BACKBONE
# ================================
vit_model = vit.vit_b16(
    image_size=img_size,
    pretrained=True,
    include_top=False,
    pretrained_top=False
)

# Get CLS token output
x = vit_model.output
x = layers.BatchNormalization()(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(0.4)(x)
output = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs=vit_model.input, outputs=output)

print("✅ Fresh model architecture built.")

# ================================
# LOAD PREVIOUS WEIGHTS
# ================================
model.load_weights("vit_contrastive_improved.keras")

print("✅ Previous weights loaded successfully.")

# ================================
# UNFREEZE LAST BLOCKS
# ================================
for layer in model.layers[-40:]:
    layer.trainable = True

for layer in model.layers[:-40]:
    layer.trainable = False

# ================================
# COMPILE
# ================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ================================
# CALLBACKS
# ================================
callbacks = [
    EarlyStopping(patience=4, restore_best_weights=True),
    ReduceLROnPlateau(patience=2, factor=0.3, verbose=1),
    ModelCheckpoint("vit_best_weights.keras", save_best_only=True)
]

print("\n========== FINAL FINE-TUNING ==========")

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=8,
    callbacks=callbacks
)

model.save("vit_final_model.keras")

print("\n🔥 Final fine-tuning complete.")