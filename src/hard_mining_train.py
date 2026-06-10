import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ================================
# LOAD TRAIN EMBEDDINGS
# ================================
embeddings = np.load("train_embeddings.npy")
labels = np.load("train_labels.npy")

print("Embeddings shape:", embeddings.shape)

# ================================
# FILTER NV and MEL ONLY
# ================================
NV_CLASS = 5
MEL_CLASS = 4

nv_idx = np.where(labels == NV_CLASS)[0]
mel_idx = np.where(labels == MEL_CLASS)[0]

nv_embeddings = embeddings[nv_idx]
mel_embeddings = embeddings[mel_idx]

print("NV samples:", len(nv_idx))
print("MEL samples:", len(mel_idx))

# ================================
# COMPUTE SIMILARITY MATRIX
# ================================
similarity_matrix = cosine_similarity(nv_embeddings, mel_embeddings)

# For each NV sample, find most similar MEL sample
max_similarities = similarity_matrix.max(axis=1)

# Threshold for confusion
threshold = 0.75  # adjust later if needed

hard_nv_indices = nv_idx[max_similarities > threshold]

print("\nNumber of hard NV samples:", len(hard_nv_indices))

# Save hard samples
np.save("hard_nv_indices.npy", hard_nv_indices)

print("Hard samples saved.")