import tensorflow as tf
import vit_keras
from vit_keras.layers import ClassToken
from tensorflow.keras.utils import plot_model

# ------------------------------
# Recreate Focal Loss
# ------------------------------
def weighted_focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)

        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = tf.pow(1 - y_pred, gamma)

        focal = weight * cross_entropy

        return tf.reduce_mean(tf.reduce_sum(focal, axis=1))
    return loss


# ------------------------------
# Load Model
# ------------------------------
model = tf.keras.models.load_model(
    "vit_micro_finetuned.keras",
    custom_objects={
        "ClassToken": ClassToken,
        "FocalLoss": weighted_focal_loss()   # ✅ Fix for error
    },
    safe_mode=False
)

# ------------------------------
# Print Model Summary
# ------------------------------
print("\nModel Summary:\n")
model.summary()

# ------------------------------
# Save Model Architecture Image
# ------------------------------
plot_model(
    model,
    to_file="model_architecture.png",
    show_shapes=True,
    show_layer_names=True,
    dpi=200
)

print("\n✅ Model architecture saved as 'model_architecture.png'")