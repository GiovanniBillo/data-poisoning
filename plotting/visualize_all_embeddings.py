import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

base_class = 0
target_class = 8

MODEL = "HG"
DATASET = "EUROSAT"
EPS = 8.0 


def plot_feature_pca_all_classes(feat_path, class_names=None, dim=2, title=None, save_path=None):
    feats, labels, ids = pickle.load(open(feat_path, 'rb'))

    labels = np.array(labels)
    feats = np.array(feats)

    assert dim in [2, 3], "Only 2D or 3D projection supported"

    # PCA projection
    pca = PCA(n_components=dim)
    proj_feats = pca.fit_transform(feats)

    unique_labels = sorted(np.unique(labels))
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(unique_labels))]

    highlight_styles = {
        base_class: {'color': 'red', 'label': f"Base ({class_names[base_class]})", 'marker': 'o', 's': 30},
        target_class: {'color': 'black', 'label': f"Target ({class_names[target_class]})", 'marker': 'x', 's': 50}
    }

    if dim == 2:
        plt.figure(figsize=(9, 8))
        for idx, cls in enumerate(unique_labels):
            cls_mask = labels == cls
            label_name = class_names[cls] if class_names else str(cls)
            style = highlight_styles.get(cls, {})
            plt.scatter(
                proj_feats[cls_mask, 0], proj_feats[cls_mask, 1],
                label=style.get('label', label_name),
                alpha=0.5,
                s=style.get('s', 10),
                color=style.get('color', colors[idx]),
                marker=style.get('marker', '.')
            )
        plt.xlabel("PC1")
        plt.ylabel("PC2")
    else:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for idx, cls in enumerate(unique_labels):
            cls_mask = labels == cls
            label_name = class_names[cls] if class_names else str(cls)
            style = highlight_styles.get(cls, {})
            ax.scatter(
                proj_feats[cls_mask, 0], proj_feats[cls_mask, 1], proj_feats[cls_mask, 2],
                label=style.get('label', label_name),
                alpha=0.5,
                s=style.get('s', 10),
                color=style.get('color', colors[idx]),
                marker=style.get('marker', '.')
            )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")

    plt.legend(loc='best', markerscale=2)
    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

# EuroSAT class names
eurosat_classes = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]

# 2D Clean
plot_feature_pca_all_classes(
    feat_path=f"models/{MODEL}_{DATASET}_{EPS}_clean_model/clean_features.pickle",
    class_names=eurosat_classes,
    dim=2,
    title="EuroSAT PCA (2D) - Clean",
    save_path="plots/pca_clean_2d.pdf"
)

# 2D Defended
plot_feature_pca_all_classes(
    feat_path=f"models/{MODEL}_{DATASET}_{EPS}_defended_model/def_features.pickle",
    class_names=eurosat_classes,
    dim=2,
    title="EuroSAT PCA (2D) - Defended",
    save_path="plots/pca_poisoned_2d.pdf"
)

# 3D Clean
plot_feature_pca_all_classes(
    feat_path=f"models/{MODEL}_{DATASET}_{EPS}_clean_model/clean_features.pickle",
    class_names=eurosat_classes,
    dim=3,
    title="EuroSAT PCA (3D) - Clean",
    save_path="plots/pca_clean_3d.pdf"
)

# 3D Defended 
plot_feature_pca_all_classes(
    feat_path=f"models/{MODEL}_{DATASET}_{EPS}_defended_model/def_features.pickle",
    class_names=eurosat_classes,
    dim=3,
    title="EuroSAT PCA (3D) - Clean",
    save_path="plots/pca_poisoned_3d.pdf"
)
