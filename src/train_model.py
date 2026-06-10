import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from vit_keras import vit

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))

# ================================
# PATHS
# ================================
base_path = "/Users/sahajdang/Documents/dataset"
image_dir = os.path.join(base_path, "images")
csv_path = os.path.join(base_path, "classification_ready.csv")

# ================================
# LOAD DATA
# ================================
df = pd.read_csv(csv_path)
df["image"] = df["image"] + ".jpg"

# ================================
# TRAIN / VAL / TEST SPLIT
# ================================
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

print("\nTrain:", len(train_df))
print("Val:", len(val_df))
print("Test:", len(test_df))

# ================================
# IMAGE SETTINGS
# ================================
img_size = 384
batch_size = 8

# ================================
# DATA AUGMENTATION
# ================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest"
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

test_gen = val_datagen.flow_from_dataframe(
    test_df,
    directory=image_dir,
    x_col="image",
    y_col="label",
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

num_classes = len(train_gen.class_indices)
print("\nClasses:", train_gen.class_indices)

# ================================
# WEIGHTED FOCAL LOSS
# ================================
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_df["label"]),
    y=train_df["label"]
)

alpha = tf.constant(class_weights, dtype=tf.float32)

def weighted_focal_loss(gamma=2.0):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = tf.pow(1 - y_pred, gamma)
        focal = alpha * weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(focal, axis=1))
    return loss

# ================================
# BUILD ViT MODEL (FIXED)
# ================================
vit_model = vit.vit_b16(
    image_size=img_size,
    pretrained=True,
    include_top=False,
    pretrained_top=False
)

vit_model.trainable = False

x = vit_model.output   # <-- already (None, 768)

x = layers.BatchNormalization()(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)
output = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs=vit_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=weighted_focal_loss(),
    metrics=["accuracy"]
)

model.summary()

# ================================
# CALLBACKS
# ================================
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint("best_vit_model.keras", save_best_only=True)
]

# ================================
# STAGE 1 TRAINING
# ================================
print("\n========== STAGE 1 ==========")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    callbacks=callbacks
)

# ================================
# STAGE 2 FINE-TUNING
# ================================
print("\n========== STAGE 2 ==========")

vit_model.trainable = True
for layer in vit_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=weighted_focal_loss(),
    metrics=["accuracy"]
)

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=callbacks
)

# ================================
# EVALUATION
# ================================
print("\n========== FINAL EVALUATION ==========")

test_loss, test_acc = model.evaluate(test_gen)
print("Test Accuracy:", test_acc)

pred_probs = model.predict(test_gen)
pred_labels = np.argmax(pred_probs, axis=1)
true_labels = test_gen.classes

print("\nClassification Report:")
print(classification_report(true_labels, pred_labels,
      target_names=list(train_gen.class_indices.keys())))

print("\nBalanced Accuracy:",
      balanced_accuracy_score(true_labels, pred_labels))

print("\nConfusion Matrix:")
print(confusion_matrix(true_labels, pred_labels))

model.save("vit_skin_model.keras")

print("\nViT Training Complete.")