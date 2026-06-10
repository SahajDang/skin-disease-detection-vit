import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from vit_keras import visualize

# ================================
# CONFIG
# ================================
img_size = 384
model_path = "vit_skin_model.keras"
image_dir = "/Users/sahajdang/Documents/dataset/images"
csv_path = "/Users/sahajdang/Documents/dataset/classification_ready.csv"

save_base = "results"
correct_dir = os.path.join(save_base, "correct")
wrong_dir = os.path.join(save_base, "wrong")

os.makedirs(correct_dir, exist_ok=True)
os.makedirs(wrong_dir, exist_ok=True)

# ================================
# LOAD MODEL
# ================================
model = tf.keras.models.load_model(
    model_path,
    compile=False,
    safe_mode=False
)

print("Model Loaded.")

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

test_df = test_df.reset_index(drop=True)

class_names = sorted(test_df["label"].unique())

print("Processing test images...")

# ================================
# PROCESS ALL TEST IMAGES
# ================================
for i in range(len(test_df)):

    img_name = test_df.loc[i, "image"]
    true_label = test_df.loc[i, "label"]

    img_path = os.path.join(image_dir, img_name)

    # Load image
    img = image.load_img(img_path, target_size=(img_size, img_size))
    img_array = image.img_to_array(img)
    input_tensor = np.expand_dims(img_array / 255.0, axis=0)

    # Predict
    preds = model.predict(input_tensor, verbose=0)
    pred_index = np.argmax(preds)
    pred_label = class_names[pred_index]
    confidence = np.max(preds)

    # Attention map
    attention_map = visualize.attention_map(model, img_array)

    # Create figure
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(img_array.astype("uint8"))
    plt.title(f"True: {true_label}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(attention_map)
    plt.title(f"Pred: {pred_label}\nConf: {confidence:.2f}")
    plt.axis("off")

    # Decide folder
    if pred_label == true_label:
        save_path = os.path.join(
            correct_dir,
            f"{img_name}_CORRECT_{true_label}.png"
        )
    else:
        save_path = os.path.join(
            wrong_dir,
            f"{img_name}_TRUE_{true_label}_PRED_{pred_label}.png"
        )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    if i % 50 == 0:
        print(f"Processed {i}/{len(test_df)}")

print("\nAll attention maps saved successfully.")
print(f"Correct saved in: {correct_dir}")
print(f"Wrong saved in: {wrong_dir}")