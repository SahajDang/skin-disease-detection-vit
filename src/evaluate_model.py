import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras

keras.config.enable_unsafe_deserialization()

# Image settings
IMG_SIZE = (224,224)
BATCH_SIZE = 16

test_dir = "dataset/test"   # change if your test folder name is different

datagen = ImageDataGenerator(rescale=1./255)

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# -------- CHANGE MODEL NAME HERE --------
MODEL_PATH = "balanced_efficientnet_model.keras"

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
    safe_mode=False
)

print("✅ Model Loaded")

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

loss, acc = model.evaluate(test_generator)

print("\n🔥 TEST Accuracy:", acc)