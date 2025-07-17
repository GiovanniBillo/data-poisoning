# -*- coding: utf-8 -*-
"""try_visualize.ipynb
What i am trying to do with this file is visualize the embeddings and the poisons.
However I am getting the same problems cyclically: there seems to be some issue with the shapes of the embeddings still.
This is the error I get
matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 128 is different from 256)

Or even worse, I get that numpy_.core.numeric is not available, so i can't do anything. 
"""

import sys
import os

# Add project root (one level up from this script's directory)
# project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
# project_root = os.path.abspath(os.path.join(os.getcwd()))
# sys.path.insert(0, project_root)

# Confirm it's now importable
# print("Project root added to path:", project_root)

import torch
from plotting.featurespace_visualizations import *
import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print(np.__version__)

use_cuda = torch.cuda.is_available()
print(use_cuda)
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

base_class = 8
target_class = 0

# poison_ids = pickle.load( open( "models/HG_EUROSAT_16.0_poison_indices.pickle", "rb" ) ).cpu().numpy()
poison_ids = pickle.load( open( "models/HG_CIFAR100_16.0_poison_indices.pickle", "rb" ) ).cpu().numpy()
poison_ids = list(map(str, poison_ids))

# ! cp -r models/model_defended/defended_model/ models/model_undefended/
# ! cp -r models/model_undefended/clean_model models/model_defended/

# ! ls /content/data-poisoning/models/HG_CIFAR100_16.0_clean_model/

generate_plots('models/HG_CIFAR100_16.0_',"gm_fromscratch",genplot_centroid_prob_2d, target_class, base_class,poison_ids, device)

generate_plots('models/HG_CIFAR100_16.0_',"gm_fromscratch",genplot_centroid_prob_3d, target_class, base_class,poison_ids, device)

# generate_plots('models/model',"gm_fromscratch",generate_plot_centroid, target_class, base_class,poison_ids, device)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA

# # Helper function to get activations
# def get_activations(model, input_tensor, layer_index):
#     activations = []
#     def hook(module, input, output):
#         activations.append(output.detach().cpu())

#     # Find the layer by index (this is a simplified approach)
#     # A more robust approach would involve naming layers and accessing them directly
#     hook_handle = None
#     current_layer_index = 0
#     for name, layer in model.named_modules():
#         if isinstance(layer, (nn.Conv2d, nn.Linear, nn.ReLU, nn.MaxPool2d, nn.BatchNorm2d, nn.AdaptiveAvgPool2d)):
#             if current_layer_index == layer_index:
#                 hook_handle = layer.register_forward_hook(hook)
#                 break
#             current_layer_index += 1

#     if hook_handle is None:
#         print(f"Warning: Could not find layer at index {layer_index}")
#         return None

#     with torch.no_grad():
#         model(input_tensor)

#     hook_handle.remove()
#     return activations[0] if activations else None


# # Helper function for softmax
# def softmax(x, theta = 1.0, axis = None):
#     """
#     Compute the softmax of each element along an axis of a numpy array.
#     """
#     y = np.atleast_2d(x)
#     if axis is None:
#         axis = y.ndim - 1
#     y = y - np.expand_dims(np.max(y, axis = axis), axis)
#     y = np.exp(y * theta)
#     ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
#     p = y / ax_sum
#     return p

# def bypass_last_layer(model):
#     """Hacky way of separating features and classification head for many models.
#     Patch this function if problems appear.

#     Modified to handle Sequential containing Linear layers.
#     """
#     layer_cake = list(model.children())

#     # Iterate backwards to find the last layer with weights (Linear or Conv)
#     last_layer = None
#     headless_layers = []
#     found_last = False

#     for layer in reversed(layer_cake):
#         if isinstance(layer, (nn.Linear, nn.Conv2d)):
#             if not found_last:
#                 last_layer = layer
#                 found_last = True
#             else:
#                 headless_layers.insert(0, layer)
#         elif isinstance(layer, nn.Sequential):
#             # If it's a sequential, look inside for the last linear/conv layer
#             seq_children = list(layer.children())
#             found_in_seq = False
#             temp_seq_layers = []
#             for sub_layer in reversed(seq_children):
#                 if isinstance(sub_layer, (nn.Linear, nn.Conv2d)):
#                     if not found_last:
#                         last_layer = sub_layer
#                         found_last = True
#                         found_in_seq = True
#                     else:
#                         temp_seq_layers.insert(0, sub_layer)
#                 else:
#                      temp_seq_layers.insert(0, sub_layer)

#             if temp_seq_layers:
#                  headless_layers.insert(0, nn.Sequential(*temp_seq_layers))
#         else:
#              headless_layers.insert(0, layer)

#     if last_layer is None:
#         raise ValueError("Could not find a layer with weights in the model.")

#     headless_model = torch.nn.Sequential(*headless_layers, torch.nn.Flatten())

#     return headless_model, last_layer


# def generate_plot_centroid(feat_path, model_path, target_class, base_class, poison_ids, title, device):
#     print(f"FEAT PATH: {feat_path}")
#     print(f"MODEL PATH: {model_path}")

#     try:
#         with open(feat_path, 'rb') as f:
#             feat_dict = pickle.load(f)
#     except FileNotFoundError:
#         print(f"Error: Feature file not found at {feat_path}")
#         return
#     except Exception as e:
#         print(f"Error loading feature file: {e}")
#         return

#     all_ops = feat_dict['ops']
#     all_labels = feat_dict['labels']
#     all_indices = feat_dict['indices']

#     # Convert indices to strings for comparison
#     all_indices_str = list(map(str, all_indices))

#     # Separate poison and clean features/labels
#     poison_ops = []
#     poison_labels = []
#     clean_ops = []
#     clean_labels = []
#     clean_base_ops = []
#     clean_base_labels = []
#     clean_target_ops = []
#     clean_target_labels = []


#     for i, index in enumerate(all_indices_str):
#         if index in poison_ids:
#             poison_ops.append(all_ops[i])
#             poison_labels.append(all_labels[i])
#         else:
#             clean_ops.append(all_ops[i])
#             clean_labels.append(all_labels[i])
#             if all_labels[i] == base_class:
#                 clean_base_ops.append(all_ops[i])
#                 clean_base_labels.append(all_labels[i])
#             elif all_labels[i] == target_class:
#                 clean_target_ops.append(all_ops[i])
#                 clean_target_labels.append(all_labels[i])


#     poison_ops = np.array(poison_ops)
#     poison_labels = np.array(poison_labels)
#     clean_ops = np.array(clean_ops)
#     clean_labels = np.array(clean_labels)
#     clean_base_ops = np.array(clean_base_ops)
#     clean_base_labels = np.array(clean_base_labels)
#     clean_target_ops = np.array(clean_target_ops)
#     clean_target_labels = np.array(clean_target_labels)


#     # Calculate centroids
#     centroid_base = np.mean(clean_base_ops, axis=0)
#     centroid_target = np.mean(clean_target_ops, axis=0)

#     # Project all data onto the line connecting the centroids
#     v = centroid_target - centroid_base
#     v_normalized = v / np.linalg.norm(v)

#     projected_clean = np.dot(clean_ops - centroid_base, v_normalized)
#     projected_poison = np.dot(poison_ops - centroid_base, v_normalized)


#     # Prepare data for plotting
#     data = pd.DataFrame({
#         'Projected Feature': np.concatenate([projected_clean, projected_poison]),
#         'Type': ['Clean'] * len(projected_clean) + ['Poison'] * len(projected_poison),
#         'Label': np.concatenate([clean_labels, poison_labels])
#     })

#     # Plotting
#     plt.figure(figsize=(10, 6))
#     sns.histplot(data=data, x='Projected Feature', hue='Type', kde=True, stat='density', common_norm=False)
#     plt.title(f'Feature Projection onto Centroid Difference Vector ({title})')
#     plt.xlabel('Projection Value')
#     plt.ylabel('Density')
#     plt.show()


# def genplot_centroid_prob_2d(feat_path, model_path, target_class, base_class, poison_ids, title, device):
#     print(f"FEAT PATH:", feat_path)
#     print(f"MODEL PATH:", model_path)

#     try:
#         with open(feat_path, 'rb') as f:
#             feat_dict = pickle.load(f)
#     except FileNotFoundError:
#         print(f"Error: Feature file not found at {feat_path}")
#         return
#     except Exception as e:
#         print(f"Error loading feature file: {e}")
#         return

#     ops_all = feat_dict['ops']
#     labels_all = feat_dict['labels']
#     indices_all = feat_dict['indices']

#     # Convert indices to strings for comparison
#     indices_all_str = list(map(str, indices_all))

#     # Separate poison and clean data
#     ops_poison = []
#     labels_poison = []
#     ops_clean = []
#     labels_clean = []

#     for i, index in enumerate(indices_all_str):
#         if index in poison_ids:
#             ops_poison.append(ops_all[i])
#             labels_poison.append(labels_all[i])
#         else:
#             ops_clean.append(ops_all[i])
#             labels_clean.append(labels_all[i])

#     ops_poison = np.array(ops_poison)
#     labels_poison = np.array(labels_poison)
#     ops_clean = np.array(ops_clean)
#     labels_clean = np.array(labels_clean)

#     # Load model
#     try:
#         model = torch.load(model_path, map_location=device)
#         if isinstance(model, nn.DataParallel):
#             model = model.module
#         model.eval()
#     except FileNotFoundError:
#         print(f"Error: Model file not found at {model_path}")
#         return
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         return

#     model.to(device)
#     headless_model, last_layer  = bypass_last_layer(model)

#     # Ensure last_layer has weights
#     if not hasattr(last_layer, 'weight') or last_layer.weight is None:
#          print(f"Error: The identified last layer does not have a 'weight' attribute.")
#          return

#     last_layer_weights = last_layer.weight.detach().cpu().numpy()
#     logit_matrix = np.matmul(ops_all, last_layer_weights.T)
#     softmax_scores = softmax(logit_matrix, theta = 1, axis = 1)

#     # Probability of target class
#     prob_target = softmax_scores[:, target_class]

#     # Project features onto the weight vector difference of target and base classes
#     # This is a simplification, ideally we should use centroids in feature space
#     # but this aligns with the 2D probability visualization approach
#     weight_diff = last_layer_weights[target_class] - last_layer_weights[base_class]
#     # Normalize the weight difference vector
#     weight_diff_normalized = weight_diff / (np.linalg.norm(weight_diff) + 1e-8) # Add epsilon for stability

#     # Project features onto this normalized weight difference vector
#     projected_features = np.dot(ops_all, weight_diff_normalized)

#     # Prepare data for plotting
#     data = pd.DataFrame({
#         'Projected Feature (Weight Diff)': projected_features,
#         f'P(Class {target_class})': prob_target,
#         'Type': ['Clean' if str(idx) not in poison_ids else 'Poison' for idx in indices_all],
#         'Label': labels_all
#     })

#     # Separate data by type for plotting
#     data_clean = data[data['Type'] == 'Clean']
#     data_poison = data[data['Type'] == 'Poison']

#     plt.figure(figsize=(10, 6))

#     # Plot clean data with different colors for base and target classes
#     sns.scatterplot(data=data_clean, x='Projected Feature (Weight Diff)', y=f'P(Class {target_class})', hue='Label', palette='viridis', label='Clean')

#     # Plot poison data
#     sns.scatterplot(data=data_poison, x='Projected Feature (Weight Diff)', y=f'P(Class {target_class})', color='red', label='Poison', s=50, marker='X')


#     plt.title(f'Feature Projection vs. P(Class {target_class}) ({title})')
#     plt.xlabel('Projected Feature onto (Target - Base) Weight Vector')
#     plt.ylabel(f'Probability of Class {target_class}')
#     plt.legend()
#     plt.grid(True)
#     plt.show()


# def genplot_centroid_prob_3d(feat_path, model_path, target_class, base_class, poison_ids, title, device):
#     print(f"FEAT PATH:", feat_path)
#     print(f"MODEL PATH:", model_path)

#     try:
#         with open(feat_path, 'rb') as f:
#             feat_dict = pickle.load(f)
#     except FileNotFoundError:
#         print(f"Error: Feature file not found at {feat_path}")
#         return
#     except Exception as e:
#         print(f"Error loading feature file: {e}")
#         return


#     ops_all = feat_dict['ops']
#     labels_all = feat_dict['labels']
#     indices_all = feat_dict['indices']

#     # Convert indices to strings for comparison
#     indices_all_str = list(map(str, indices_all))

#     # Separate poison and clean data
#     ops_poison = []
#     labels_poison = []
#     ops_clean = []
#     labels_clean = []

#     for i, index in enumerate(indices_all_str):
#         if index in poison_ids:
#             ops_poison.append(ops_all[i])
#             labels_poison.append(labels_all[i])
#         else:
#             ops_clean.append(ops_all[i])
#             labels_clean.append(labels_all[i])

#     ops_poison = np.array(ops_poison)
#     labels_poison = np.array(labels_poison)
#     ops_clean = np.array(ops_clean)
#     labels_clean = np.array(labels_clean)

#     # Load model
#     try:
#         model = torch.load(model_path, map_location=device)
#         if isinstance(model, nn.DataParallel):
#             model = model.module
#         model.eval()
#     except FileNotFoundError:
#         print(f"Error: Model file not found at {model_path}")
#         return
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         return

#     model.to(device)
#     headless_model, last_layer  = bypass_last_layer(model)

#     # Ensure last_layer has weights
#     if not hasattr(last_layer, 'weight') or last_layer.weight is None:
#          print(f"Error: The identified last layer does not have a 'weight' attribute.")
#          return

#     last_layer_weights = last_layer.weight.detach().cpu().numpy()
#     logit_matrix = np.matmul(ops_all, last_layer_weights.T)
#     softmax_scores = softmax(logit_matrix, theta = 1, axis = 1)

#     # Probabilities for base and target classes
#     prob_base = softmax_scores[:, base_class]
#     prob_target = softmax_scores[:, target_class]

#     # Prepare data for plotting
#     data = pd.DataFrame({
#         f'P(Class {base_class})': prob_base,
#         f'P(Class {target_class})': prob_target,
#         'Type': ['Clean' if str(idx) not in poison_ids else 'Poison' for idx in indices_all],
#         'Label': labels_all
#     })

#     # Separate data by type for plotting
#     data_clean = data[data['Type'] == 'Clean']
#     data_poison = data[data['Type'] == 'Poison']


#     fig = plt.figure(figsize=(12, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     # Plot clean data
#     scatter_clean = ax.scatter(data_clean[f'P(Class {base_class})'], data_clean[f'P(Class {target_class})'], data_clean['Label'], c=data_clean['Label'], cmap='viridis', label='Clean')

#     # Plot poison data
#     scatter_poison = ax.scatter(data_poison[f'P(Class {base_class})'], data_poison[f'P(Class {target_class})'], data_poison['Label'], color='red', label='Poison', marker='X', s=100)


#     ax.set_xlabel(f'Probability of Class {base_class}')
#     ax.set_ylabel(f'Probability of Class {target_class}')
#     ax.set_zlabel('True Label')
#     ax.set_title(f'Probability Space Visualization ({title})')

#     # Create legend handles manually
#     legend_elements = [
#         plt.Line2D([0], [0], marker='o', color='w', label='Clean', markerfacecolor='gray', markersize=10),
#         plt.Line2D([0], [0], marker='X', color='w', label='Poison', markerfacecolor='red', markersize=10)
#     ]

#     # Add legend for labels (classes)
#     unique_labels = sorted(data['Label'].unique())
#     cmap = plt.cm.viridis
#     norm = plt.Normalize(min(unique_labels), max(unique_labels))

#     for label in unique_labels:
#         color = cmap(norm(label))
#         legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label=f'Label {label}', markerfacecolor=color, markersize=10))


#     ax.legend(handles=legend_elements, loc='best')

#     plt.show()


# def generate_plots(main_path, model_name, plot_function, target_class, base_class, poison_ids, device):
#     print(f"MAIN PATH: {main_path}")

#     # Construct file paths based on the main_path and model_name
#     # Assumes a standard directory structure within main_path
#     feat_path_clean = os.path.join(main_path, f"{model_name}_clean_model", "clean_features.pickle")
#     model_path_clean = os.path.join(main_path, f"{model_name}_clean_model", "clean.pth")

#     feat_path_poisoned = os.path.join(main_path, f"{model_name}_poisoned_model", "poisoned_features.pickle")
#     model_path_poisoned = os.path.join(main_path, f"{model_name}_poisoned_model", "poisoned.pth")

#     # Clean model plot
#     if os.path.exists(feat_path_clean) and os.path.exists(model_path_clean):
#         print(f"Generating plot for Clean model...")
#         plot_function(feat_path_clean, model_path_clean, target_class, base_class, poison_ids,
#                             model_name + " (Clean)", device)
#     else:
#         print(f"Clean model files not found at {feat_path_clean} and {model_path_clean}. Skipping clean plot.")

#     # Poisoned model plot
#     if os.path.exists(feat_path_poisoned) and os.path.exists(model_path_poisoned):
#         print(f"Generating plot for Poisoned model...")
#         plot_function(feat_path_poisoned, model_path_poisoned, target_class, base_class, poison_ids,
#                             model_name + " (Poisoned)", device)
#     else:
#         print(f"Poisoned model files not found at {feat_path_poisoned} and {model_path_poisoned}. Skipping poisoned plot.")
