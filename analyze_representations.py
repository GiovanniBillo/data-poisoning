import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
import torch
from tqdm import tqdm
import time
import os

LIMIT= 10000
print("[INFO] Starting script...")
print(f"[INFO] limiting data considered to LIMIT={LIMIT}...")

# === Load embeddings ===
print("[INFO] Loading clean embeddings...")
clean_fc = torch.load("embeddings/embeddings_CLEAN_epoch20.pth", map_location="cpu")
print(f"[INFO] Loaded {len(clean_fc)} batches")

# === Concatenate 'fc_out' from all batches ===
print("[INFO] Extracting 'fc_out' from all batches...")
all_fc_out = [entry['fc_out'] for entry in clean_fc]
reshaped_fc = torch.cat(all_fc_out, dim=0).numpy()
reshaped_fc = reshaped_fc[:LIMIT] 
print(f"[INFO] Combined embedding shape: {reshaped_fc.shape}")

# === UMAP ===
print("[INFO] Running UMAP...")
start = time.time()
reducer = UMAP(random_state=42, verbose=True)
embedding_2d = reducer.fit_transform(reshaped_fc)
end = time.time()
print(f"[INFO] UMAP finished in {end - start:.2f} seconds")
print(f"[INFO] UMAP output shape: {embedding_2d.shape}")

# === Load labels ===
print("[INFO] Loading training labels...")
train_data = torch.load("embeddings/train_samples.pth", map_location="cpu")
train_labels = train_data['labels'][:LIMIT].numpy()

# === Sanity check ===
if embedding_2d.shape[0] != len(train_labels):
    raise ValueError(f"Mismatch: {embedding_2d.shape[0]} embeddings vs {len(train_labels)} labels")

# === Plot ===
print("[INFO] Plotting...")
plt.figure(figsize=(10, 8))
plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=train_labels, cmap='Spectral', s=5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
plt.title('UMAP of Embeddings (FC Layer)', fontsize=24)
plt.tight_layout()
plt.show()
plt.savefig("representations_clean_plot.png")

# === Load poisoned embeddings ===
print("[INFO] Loading poisoned embeddings...")
poisoned_fc = torch.load("embeddings/embeddings_POISONED_epoch20.pth", map_location="cpu")
print(f"[INFO] Loaded {len(poisoned_fc)} batches")

# === Concatenate 'fc_out' from all (poisoned) batches ===
print("[INFO] Extracting 'poisoned_fc_out' from all batches...")
all_poisoned_fc_out = [entry['fc_out'] for entry in poisoned_fc]
reshaped_poisoned_fc = torch.cat(all_poisoned_fc_out, dim=0).numpy()
reshaped_poisoned_fc = reshaped_poisoned_fc[:LIMIT] 
print(f"[INFO] Combined embedding shape: {reshaped_poisoned_fc.shape}")

# === UMAP ===
print("[INFO] Running UMAP...")
start = time.time()
reducer = UMAP(random_state=42, verbose=True)
embedding_2d = reducer.fit_transform(reshaped_poisoned_fc)
end = time.time()
print(f"[INFO] UMAP finished in {end - start:.2f} seconds")
print(f"[INFO] UMAP output shape: {embedding_2d.shape}")

# === Load labels ===
print("[INFO] Loading training labels...")
train_data = torch.load("embeddings/train_samples.pth", map_location="cpu")
train_labels = train_data['labels'][:LIMIT].numpy()

# === Sanity check ===
if embedding_2d.shape[0] != len(train_labels):
    raise ValueError(f"Mismatch: {embedding_2d.shape[0]} embeddings vs {len(train_labels)} labels")

# === Plot ===
print("[INFO] Plotting...")
plt.figure(figsize=(10, 8))
plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=train_labels, cmap='Spectral', s=5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
plt.title('UMAP of Embeddings (FC Layer)', fontsize=24)
plt.tight_layout()
plt.show()
plt.savefig("representations_poisoned_plot.png")


print("[INFO] Representation analysis completed successfully.")
# # | Layer            | Output Shape (for 64×64 input) | # Channels / Units |
# # | ---------------- | ------------------------------ | ------------------ |
# # | `initial_layer`  | `[64, 64, 64]`                 | 64 channels        |
# # | `down_block1`    | `[128, 32, 32]`                | 128 channels       |
# # | `down_block2`    | `[256, 16, 16]`                | 256 channels       |
# # | `bottleneck`     | `[256, 16, 16]`                | 256 channels       |
# # | `pool + flatten` | `[256]`                        | 256 units (1D)     |
# # | `fc[0]`          | Linear(256 → 128)              | 128 units          |
# # | `fc[3]`          | Linear(128 → num\_classes)     | N (e.g., 10 units) |

# clean_fc = torch.load("embeddings/embeddings_CLEAN_epoch20.pth", map_location="cpu")
# print(f"loaded clean_fc with {len(clean_fc)}")
# print("[INFO] Extracting 'fc_out' tensor and converting to NumPy...")

# reshaped_fc = clean_fc[0]['fc_out'].numpy()
# print(f"[INFO] Embedding tensor shape: {reshaped_fc.shape}")

# # === Run UMAP ===
# print("[INFO] Starting UMAP dimensionality reduction...")
# start = time.time()
# reducer_fc = UMAP(random_state=42, verbose=True)
# reducer_fc.fit(reshaped_fc)
# embedding_fc = reducer_fc.transform(reshaped_fc)
# end = time.time()
# print(f"[INFO] UMAP finished in {end - start:.2f} seconds")
# assert np.all(embedding_fc == reducer_fc.embedding_)
# print(f"[INFO] UMAP embedding shape: {embedding_fc.shape}")

# # === Load labels ===
# print("[INFO] Loading training labels...")
# directory = "embeddings"
# train_filename = "train_samples.pth"
# train_data_path =  os.path.join(directory, train_filename)
# data_train = torch.load(train_data_path, map_location="cpu")
# train_imgs, train_labels = data_train['images'], data_train['labels']
# train_labels_np = train_labels.numpy()

# print("[INFO] Loading validation labels (to compare vs POISONED)...")
# directory = "embeddings"
# valid_filename = "valid_samples.pth"
# valid_data_path =  os.path.join(directory, valid_filename)
# data_valid = torch.load(valid_data_path, map_location="cpu")
# valid_imgs, valid_labels = data_valid['images'], data_valid['labels']
# valid_labels_np = valid_labels.numpy()

# # === Plot UMAP output ===
# print("[INFO] Plotting UMAP projection...")
# plt.figure(figsize=(10, 8))
# plt.scatter(embedding_fc[:len(train_labels_np), 0],
#             embedding_fc[:len(train_labels_np), 1],
#             c=train_labels_np, cmap='Spectral', s=5)
# plt.gca().set_aspect('equal', 'datalim')
# plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
# plt.title('UMAP projection of the final FC layer', fontsize=24)
# plt.tight_layout()
# plt.show()


