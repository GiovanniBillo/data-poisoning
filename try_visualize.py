# -*- coding: utf-8 -*-
"""try_visualize.ipynb
What i am trying to do with this file is visualize the embeddings and the poisons.
However I am getting the same problems cyclically: there seems to be some issue with the shapes of the embeddings still.
This is the error I get
matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 128 is different from 256)

Or even worse, I get that numpy_.core.numeric is not available, so i can't do anything. 
"""
MODEL = "HG"
DATASET = "CIFAR100"
EPS = 16.0

import sys
import os

import torch
from plotting.featurespace_visualizations import *
import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print(f"Using numpy version {np.__version__}")

use_cuda = torch.cuda.is_available()
print("Is cuda GPU available? ", use_cuda)
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# TODO: setup with actually meaningful classes
base_class = 8
target_class = 0

# poison_ids = pickle.load( open( "models/HG_EUROSAT_16.0_poison_indices.pickle", "rb" ) ).cpu().numpy()
poison_ids = pickle.load( open( f"models/{MODEL}_{DATASET}_{EPS}_poison_indices.pickle", "rb" ) ).cpu().numpy()
poison_ids = list(map(str, poison_ids))

# ! cp -r models/model_defended/defended_model/ models/model_undefended/
# ! cp -r models/model_undefended/clean_model models/model_defended/

# ! ls /content/data-poisoning/models/HG_CIFAR100_16.0_clean_model/

generate_plots(f'models/{MODEL}_{DATASET}_{EPS}_',"gm_fromscratch",genplot_centroid_prob_2d, target_class, base_class,poison_ids, device)

generate_plots(f'models/{MODEL}_{DATASET}_{EPS}_',"gm_fromscratch",genplot_centroid_prob_3d, target_class, base_class,poison_ids, device)

