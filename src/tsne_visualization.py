import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# ================================
# LOAD EMBEDDINGS
# ================================
embeddings = np.load("vit_embeddings.npy")
labels = np.load("vit_labels.npy")

# Your class mapping
class_names = ['AKIEC','BCC','BKL','DF','MEL','NV','VASC']

# Get index of MEL and NV
mel_index = class_names.index("MEL")
nv_index = class_names.index("NV")

# ================================
# FILTER ONLY NV & MEL
# ================================
mask = (labels == mel_index) | (labels == nv_index)

filtered_embeddings = embeddings[mask]
filtered_labels = labels[mask]

print("Filtered shape:", filtered_embeddings.shape)

# ================================
# APPLY T-SNE
# ================================
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
embeddings_2d = tsne.fit_transform(filtered_embeddings)

# ================================
# PLOT
# ================================
plt.figure(figsize=(8,6))

for label_value, label_name in [(mel_index, "MEL"), (nv_index, "NV")]:
    indices = filtered_labels == label_value
    plt.scatter(
        embeddings_2d[indices, 0],
        embeddings_2d[indices, 1],
        label=label_name,
        alpha=0.6
    )

plt.legend()
plt.title("t-SNE Visualization (NV vs MEL)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()