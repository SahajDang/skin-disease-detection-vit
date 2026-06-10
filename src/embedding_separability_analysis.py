import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_distances

# ================================
# LOAD TRAIN EMBEDDINGS
# ================================
embeddings = np.load("train_embeddings.npy")
labels = np.load("train_labels.npy")

print("Embeddings shape:", embeddings.shape)
print("Unique classes:", np.unique(labels))

# ================================
# NORMALIZE EMBEDDINGS
# ================================
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings = embeddings / norms

# ================================
# INTRA-CLASS DISTANCE
# ================================
intra_distances = []

for cls in np.unique(labels):
    cls_embeddings = embeddings[labels == cls]
    
    if len(cls_embeddings) > 1:
        dists = cosine_distances(cls_embeddings)
        intra_distances.append(np.mean(dists))

avg_intra = np.mean(intra_distances)

# ================================
# INTER-CLASS DISTANCE
# ================================
inter_distances = []

classes = np.unique(labels)

for i in range(len(classes)):
    for j in range(i + 1, len(classes)):
        emb_i = embeddings[labels == classes[i]]
        emb_j = embeddings[labels == classes[j]]
        
        dists = cosine_distances(emb_i, emb_j)
        inter_distances.append(np.mean(dists))

avg_inter = np.mean(inter_distances)

# ================================
# SILHOUETTE SCORE
# ================================
sil_score = silhouette_score(embeddings, labels)

# ================================
# DAVIES-BOULDIN SCORE
# ================================
db_score = davies_bouldin_score(embeddings, labels)

# ================================
# RESULTS
# ================================
print("\n==============================")
print("Embedding Separability Report")
print("==============================")

print(f"Average Intra-class Distance: {avg_intra:.4f}")
print(f"Average Inter-class Distance: {avg_inter:.4f}")
print(f"Inter/Intra Ratio: {avg_inter / avg_intra:.4f}")

print("\nSilhouette Score:", round(sil_score, 4))
print("Davies-Bouldin Score:", round(db_score, 4))

print("\nInterpretation Guide:")
print("• Lower intra-class distance = tighter clusters")
print("• Higher inter-class distance = better separation")
print("• Higher silhouette (closer to 1) = better clustering")
print("• Lower DB score = better cluster structure")