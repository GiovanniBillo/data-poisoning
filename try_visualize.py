# -*- coding: utf-8 -*-
"""try_visualize.ipynb
Directly visualize poisoning
"""
MODEL = "HG"
DATASET = "EUROSAT"
EPS = 16.0 

import sys
import os

import torch
from plotting.featurespace_visualizations import *
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

print(f"Using numpy version {np.__version__}")

use_cuda = torch.cuda.is_available()
print("Is cuda GPU available? ", use_cuda)
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Set the class you actually want to visualize

# | Index | Class Name           |
# | ----- | -------------------- |
# | 0     | AnnualCrop           |
# | 1     | Forest               |
# | 2     | HerbaceousVegetation |
# | 3     | Highway              |
# | 4     | Industrial           |
# | 5     | Pasture              |
# | 6     | PermanentCrop        |
# | 7     | Residential          |
# | 8     | River                |
# | 9     | SeaLake              |

base_class = 3
target_class = 0

# poison_ids = pickle.load( open( "models/HG_EUROSAT_16.0_poison_indices.pickle", "rb" ) ).cpu().numpy()
poison_ids = pickle.load( open( f"models/{MODEL}_{DATASET}_{EPS}_poison_indices.pickle", "rb" ) ).cpu().numpy()
poison_ids = list(map(str, poison_ids))
print(poison_ids)
# ! cp -r models/model_defended/defended_model/ models/model_undefended/
# ! cp -r models/model_undefended/clean_model models/model_defended/

# ! ls /content/data-poisoning/models/HG_CIFAR100_16.0_clean_model/
print("Generating CENTROID PROBABILITY (2D)")
generate_plots(f'models/{MODEL}_{DATASET}_{EPS}_',"gm_fromscratch",genplot_centroid_prob_2d, target_class, base_class,poison_ids, device)

print("Generating CENTROID PROBABILITY (3D)")
generate_plots(f'models/{MODEL}_{DATASET}_{EPS}_',"gm_fromscratch",genplot_centroid_prob_3d, target_class, base_class,poison_ids, device)

print("Generating CENTROID PLOT ")
generate_plots(f'models/{MODEL}_{DATASET}_{EPS}_',"gm_fromscratch",generate_plot_centroid, target_class, base_class,poison_ids, device)

print("Generating PCA PLOT")
generate_plots(f'models/{MODEL}_{DATASET}_{EPS}_',"gm_fromscratch",generate_plot_pca, target_class, base_class,poison_ids, device)

