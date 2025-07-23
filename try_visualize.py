# visualize_all.py
import pandas as pd
import torch
import pickle
from plotting.featurespace_visualizations import *
import os

MODEL = "HG"
DATASET = "EUROSAT"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

CLASS_TO_IDX = {
    "Annual Crop": 0,
    "Forest": 1,
    "Herbaceous Vegetation": 2,
    "Highway": 3,
    "Industrial Buildings": 4,
    "Pasture": 5,
    "Permanent Crop": 6,
    "Residential Buildings": 7,
    "River": 8,
    "Sea Lake": 9
}

# === Load param grid ===
param_df = pd.read_csv("param_grid.csv")

for _, row in param_df.iterrows():
    eps = float(row["eps"])
    poisonkey = int(row["poisonkey"])
    target_name = row["target"].strip()
    goal_name = row["goal"].strip()

    if target_name not in CLASS_TO_IDX or goal_name not in CLASS_TO_IDX:
        print(f"[WARN] Unknown class name(s): {target_name}, {goal_name}")
        continue

    # triangle should be among the blue dots in clean training, among the red ones in poisoned training (when the poison actually works). 
    # targets are originally from the blue class
    # poisons are originally from the green class
    base_class = CLASS_TO_IDX[target_name]
    target_class = CLASS_TO_IDX[goal_name]

    model_name = f"{MODEL}_{DATASET}_{eps}_{poisonkey}"
    main_path = f"models/{model_name}_"
    poison_path = f"{main_path}poison_indices.pickle"
    embeddings_save_path = "plots/embeddings_plots"

    if not os.path.exists(poison_path):
        print(f"[WARN] Poison file not found: {poison_path}")
        continue

    print(f"[INFO] Generating plots for EPS={eps}, PoisonKey={poisonkey}")
    print(f"[INFO] Base class: {target_name} -> {base_class}")
    print(f"[INFO] Target class: {goal_name} -> {target_class}")

    poison_ids = pickle.load(open(poison_path, "rb")).cpu().numpy()
    poison_ids = list(map(str, poison_ids))

    print("  → CENTROID PROBABILITY (2D)")
    generate_plots(main_path, model_name, genplot_centroid_prob_2d, target_class, base_class, poison_ids, DEVICE)

    print("  → CENTROID PROBABILITY (3D)")
    generate_plots(main_path, model_name, genplot_centroid_prob_3d, target_class, base_class, poison_ids, DEVICE)

    print("  → ALL EMBEDDINGS (2D/3D)")
    generate_all_embeddings_plots(main_path, model_name, base_class, target_class, embeddings_save_path)

    print("  → CENTROID PLOT")
    generate_plots(main_path, model_name, generate_plot_centroid, target_class, base_class, poison_ids, DEVICE)

    print("  → PCA PLOT")
    generate_plots(main_path, model_name, generate_plot_pca, target_class, base_class, poison_ids, DEVICE)

# # -*- coding: utf-8 -*-
# """try_visualize.ipynb
# Directly visualize poisoning
# """
# MODEL = "HG"
# DATASET = "EUROSAT"
# EPS = 32.0 

# import sys
# import os

# import torch
# from plotting.featurespace_visualizations import *
# import os
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt

# print(f"Using numpy version {np.__version__}")

# use_cuda = torch.cuda.is_available()
# print("Is cuda GPU available? ", use_cuda)
# device = torch.device("cuda:0" if use_cuda else "cpu")
# torch.backends.cudnn.benchmark = True

# # Set the class you actually want to visualize

# # | Index | Class Name           |
# # | ----- | -------------------- |
# # | 0     | AnnualCrop           |
# # | 1     | Forest               |
# # | 2     | HerbaceousVegetation |
# # | 3     | Highway              |
# # | 4     | Industrial           |
# # | 5     | Pasture              |
# # | 6     | PermanentCrop        |
# # | 7     | Residential          |
# # | 8     | River                |
# # | 9     | SeaLake              |

# base_class = 0
# target_class = 8

# # to better identify the plots later
# model_name = f"{MODEL}_{DATASET}_{EPS}"
# main_path = f'models/{MODEL}_{DATASET}_{EPS}_'
# # embedding_features_path = f"models/{MODEL}_{DATASET}_{EPS}_clean_model/clean_features.pickle"  
# embeddings_save_path = f"plots/embeddings_plots"
# # poison_ids = pickle.load( open( "models/HG_EUROSAT_16.0_poison_indices.pickle", "rb" ) ).cpu().numpy()
# poison_ids = pickle.load( open( f"models/{MODEL}_{DATASET}_{EPS}_poison_indices.pickle", "rb" ) ).cpu().numpy()
# poison_ids = list(map(str, poison_ids))

# print(poison_ids)

# print("Generating CENTROID PROBABILITY (2D)")
# generate_plots(main_path,model_name,genplot_centroid_prob_2d, target_class, base_class,poison_ids, device)

# print("Generating CENTROID PROBABILITY (3D)")
# generate_plots(main_path,model_name,genplot_centroid_prob_3d, target_class, base_class,poison_ids, device)

# print("Generating plots for ALL embeddings (2D and 3D)")
# generate_all_embeddings_plots(main_path, model_name, base_class, target_class, embeddings_save_path)

# print("Generating CENTROID PLOT ")
# generate_plots(main_path,model_name,generate_plot_centroid, target_class, base_class,poison_ids, device)

# print("Generating PCA PLOT")
# generate_plots(main_path,model_name,generate_plot_pca, target_class, base_class,poison_ids, device)


