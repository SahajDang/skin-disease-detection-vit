import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from vit_keras import vit
from vit_keras.layers import ClassToken, AddPositionEmbs, TransformerBlock

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))

# =====================================================
# SETTINGS
# =====================================================
base_path = "/Users/sahajdang/Documents/dataset"
image_dir = os.path.join(base_path, "images")
csv_path = os.path.join(base_path, "classification_ready.csv")

img_size = 224
batch_size = 16
epochs = 12
num_classes = 7

# =====================================================
# LOAD DATA
# =====================================================
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

# =====================================================
# CLASS WEIGHTS (IMPORTANT)
# =====================================================
labels = train_df["label"].values
class_labels = np.unique(labels)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=class_labels,
    y=labels
)

class_weights_dict = dict(zip(range(len(class_labels)), class_weights))
print("Class Weights:", class_weights_dict)

# =====================================================
# BUILD VIT MODEL (FRESH)
# =====================================================
model = vit.vit_b16(
    image_size=img_size,
    pretrained=True,
    include_top=False,
    pretrained_top=False
)

x = tf.keras.layers.Dense(512, activation="relu")(model.output)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

model = tf.keras.models.Model(inputs=model.input, outputs=output)

print("✅ Fresh ViT built.")

# =====================================================
# PARTIAL FINE-TUNING (ONLY LAST BLOCKS)
# =====================================================
for layer in model.layers:
    layer.trainable = False

for layer in model.layers[-40:]:
    layer.trainable = True

# =====================================================
# FOCAL LOSS
# =====================================================
def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1. - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1 - y_pred, gamma)
        return tf.reduce_sum(weight * ce, axis=1)
    return loss

# =====================================================
# COSINE LR SCHEDULE
# =====================================================
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-5,
    decay_steps=1000
)

optimizer = tf.keras.optimizers.Adam(lr_schedule)

model.compile(
    optimizer=optimizer,
    loss=focal_loss(),
    metrics=["accuracy"]
)

model.summary()

# =====================================================
# CALLBACKS
# =====================================================
callbacks = [
    EarlyStopping(patience=4, restore_best_weights=True),
    ModelCheckpoint("vit_best_stable.keras", save_best_only=True)
]

# =====================================================
# TRAIN
# =====================================================
print("\n========== FINAL TRAINING ==========")

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    class_weight=class_weights_dict,
    callbacks=callbacks
)

print("🔥 Training Complete.")