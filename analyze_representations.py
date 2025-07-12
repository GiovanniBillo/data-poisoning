import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
import torch
from tqdm import tqdm
import time

print("[INFO] Starting script...")

# === Load embeddings ===
print("[INFO] Loading clean embeddings...")
clean_fc = torch.load("embeddings/embeddings_CLEAN_epoch50.pth", map_location="cpu")

print("[INFO] Extracting 'fc_out' tensor and converting to NumPy...")
reshaped_fc = clean_fc[0]['fc_out'].numpy()
print(f"[INFO] Embedding tensor shape: {reshaped_fc.shape}")

# === Run UMAP ===
print("[INFO] Starting UMAP dimensionality reduction...")
start = time.time()
reducer_fc = UMAP(random_state=42, verbose=True)
reducer_fc.fit(reshaped_fc)
embedding_fc = reducer_fc.transform(reshaped_fc)
end = time.time()
print(f"[INFO] UMAP finished in {end - start:.2f} seconds")
assert np.all(embedding_fc == reducer_fc.embedding_)
print(f"[INFO] UMAP embedding shape: {embedding_fc.shape}")

# === Load labels ===
print("[INFO] Loading training labels...")
data_train = torch.load("train_samples.pth", map_location="cpu")
train_imgs, train_labels = data_train['images'], data_train['labels']
labels_np = train_labels.numpy()

# === Plot UMAP output ===
print("[INFO] Plotting UMAP projection...")
plt.figure(figsize=(10, 8))
plt.scatter(embedding_fc[:len(labels_np), 0],
            embedding_fc[:len(labels_np), 1],
            c=labels_np, cmap='Spectral', s=5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
plt.title('UMAP projection of the final FC layer', fontsize=24)
plt.tight_layout()
plt.show()

print("[INFO] Representation analysis completed successfully.")
# import numpy as np
# from sklearn.datasets import load_digits
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import umap
# import torch


# # import the last embeddings for the clean vs poisoned epochs
# clean_fc = torch.load("embeddings_CLEAN_epoch50.pth", map_location="cpu")
# poisoned_fc = torch.load("embeddings_POISONED_epoch50.pth", map_location="cpu") 

# reshaped_clean_fc = clean_fc['fc_out'].numpy()
# reshaped_poisoned_fc = poisoned_fc['fc_out'].numpy()

# reducer_fc = umap.UMAP(random_state=42)
# reducer_fc.fit(reshaped_fc)

# # reducer_h2 = umap.UMAP(random_state=43)
# # reducer_h2.fit(reshaped_h2)

# # reducer_h1 = umap.UMAP(random_state=44)
# # reducer_h1.fit(reshaped_h1)

# embedding_fc = reducer_fc.transform(reshaped_fc)
# # Verify that the result of calling transform is
# # idenitical to accessing the embedding_ attribute
# assert(np.all(embedding_fc == reducer_fc.embedding_))
# print("fc embedding shape:", embedding_fc.shape)

# data_train = torch.load("train_samples.pth", map_location="cpu")
# data_val = torch.load("valid_samples.pth", map_location="cpu")

# train_imgs, train_labels = data_train['images'], data_train['labels']
# valid_imgs, valid_labels = data_val['images'], data_val['labels']

# reshaped_clean_fc.numpy()
# reshaped_poisoned_fc.numpy()
# train_labels_concatenated = np.concatenate(train_labels)
# valid_labels_concatenated = np.concatenate(valid_labels)

# plt.scatter(embedding_fc[:len(labels_concatenated), 0], embedding_fc[:len(labels_concatenated), 1], c=labels_concatenated, cmap='Spectral', s=5)
# plt.gca().set_aspect('equal', 'datalim')
# plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
# plt.title('UMAP projection of the MNIST dataset (3rd hidden layer)', fontsize=24);
