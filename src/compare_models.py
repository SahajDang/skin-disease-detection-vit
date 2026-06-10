import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
import keras
import matplotlib.pyplot as plt

# ---- IMPORTANT FOR VIT MODELS ----
from vit_keras import vit
from vit_keras.layers import ClassToken, AddPositionEmbs, TransformerBlock

keras.config.enable_unsafe_deserialization()

# -------------------------------
# SETTINGS
# -------------------------------

IMG_SIZE = (224,224)
BATCH_SIZE = 16
TEST_DIR = "dataset/test"

# Models to compare
models = {
    "EfficientNet": "balanced_efficientnet_model.keras",
    "ViT Base": "vit_skin_model.keras",
    "ViT Contrastive": "vit_contrastive_improved.keras",
    "ViT Focal": "vit_focal_stable.keras",
    "ViT Micro FT": "vit_micro_finetuned.keras",
    "ViT Best": "vit_best_stable.keras"
}

# -------------------------------
# DATA LOADER
# -------------------------------

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_generator = datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# -------------------------------
# MODEL EVALUATION
# -------------------------------

results = {}

for name, path in models.items():

    print("\n===============================")
    print("Evaluating:", name)
    print("===============================")

    try:

        model = tf.keras.models.load_model(
            path,
            compile=False,
            safe_mode=False
        )

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        loss, acc = model.evaluate(test_generator)

        results[name] = acc
        print("Accuracy:", acc)

    except Exception as e:

        print("❌ Could not evaluate:", name)
        print(e)

# -------------------------------
# PLOT RESULTS
# -------------------------------

names = list(results.keys())
accuracies = list(results.values())

plt.figure(figsize=(10,6))

bars = plt.bar(names, accuracies)

plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.title("Model Performance Comparison")
plt.ylim(0,1)

# show numbers above bars
for i, v in enumerate(accuracies):
    plt.text(i, v+0.01, f"{v:.3f}", ha='center', fontsize=10)

plt.tight_layout()

plt.savefig("model_comparison.png", dpi=300)

plt.show()

print("\n✅ Graph saved as model_comparison.png")
print("Results:", results)