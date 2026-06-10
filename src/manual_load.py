import tensorflow as tf
import keras
keras.config.enable_unsafe_deserialization()

from vit_keras import vit
from tensorflow.keras import layers, models

img_size = 384
num_classes = 7

# ================================
# Rebuild Architecture
# ================================
vit_model = vit.vit_b16(
    image_size=img_size,
    pretrained=False,
    include_top=False,
    pretrained_top=False
)

x = vit_model.output
x = layers.BatchNormalization()(x)

# same normalization layer as training
x = layers.Lambda(
    lambda x: tf.math.l2_normalize(x, axis=1),
    output_shape=(768,),
    name="normalized_embedding"
)(x)

x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(0.4)(x)
output = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs=vit_model.input, outputs=output)

# ================================
# Load Weights Only
# ================================
model.load_weights("vit_contrastive_improved.keras")

print("✅ Model loaded successfully.")
model.summary()

# ================================
# VALIDATION CHECK
# ================================
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

base_path = "/Users/sahajdang/Documents/dataset"
image_dir = os.path.join(base_path, "images")
csv_path = os.path.join(base_path, "classification_ready.csv")

df = pd.read_csv(csv_path)
df["image"] = df["image"] + ".jpg"

_, temp_df = train_test_split(
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

datagen = ImageDataGenerator(rescale=1./255)

val_gen = datagen.flow_from_dataframe(
    val_df,
    directory=image_dir,
    x_col="image",
    y_col="label",
    target_size=(384, 384),
    batch_size=8,
    class_mode="categorical"
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

loss, acc = model.evaluate(val_gen)
print("\n🔥 Validation Accuracy:", acc)

# TEST EVALUATION
_, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["label"],
    random_state=42
)

test_gen = datagen.flow_from_dataframe(
    test_df,
    directory=image_dir,
    x_col="image",
    y_col="label",
    target_size=(384, 384),
    batch_size=8,
    class_mode="categorical",
    shuffle=False
)

loss, acc = model.evaluate(test_gen)
print("\n🔥 Test Accuracy:", acc)