import torch
import torch.nn as nn
import torch.nn.functional as F
# from forest.victims.models import ResNet, resnet_picker

# from forest.victims.HG import HG
from HG import HG

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

NUM_CLASSES=10 # modify accordingly: should be set outside or by some other means really.

# EuroSAT class names
eurosat_classes = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]

def generate_plot_centroid(feat_path,model_path,target_class,base_class, poison_ids, title, device):

    [ops_all, labels_all, ids_all] = pickle.load( open( feat_path, "rb" ) ) 
    
    
    left_labels = np.array(labels_all)[np.in1d(labels_all,[target_class,base_class])]
    left_ids = np.array(ids_all)[np.in1d(labels_all,[target_class,base_class])]
    left_ops = ops_all[np.in1d(labels_all,[target_class,base_class])]
    id2label = dict(zip(left_ids, left_labels))

    tags = []
    for i in left_ids:
        if str(i) in poison_ids:
            print("entered poison!!!")
            tags.append('poison')
        elif i == 'target':
            tags.append('target')
        elif id2label[i]== target_class:
            tags.append(str(target_class))
        else:
            tags.append(str(base_class))

    tags = np.array(tags)  
    print(np.sum(tags == str(target_class)), 
          np.sum(tags == str(base_class)), np.sum(tags == str('poison')), np.sum(tags == str('target')))
    
    basefeats = left_ops[tags == str(base_class)]
    targfeats = left_ops[tags == str(target_class)]
    basecent = np.mean(basefeats, axis=0)
    targcent = np.mean(targfeats, axis=0)
    distcent = targcent - basecent
    distcent /= np.linalg.norm(distcent)
    distcent *= -1

    basefeats_ = basefeats.dot(distcent)
    basefeats_ = np.outer(basefeats_, distcent)
    basefeats_ = basefeats - basefeats_

    targfeats_ = targfeats.dot(distcent)
    targfeats_ = np.outer(targfeats_, distcent)
    targfeats_ = targfeats - targfeats_

    a = np.concatenate([basefeats_, targfeats_])
    pca = PCA(n_components=3)
    pca.fit(np.concatenate([basefeats_, targfeats_]))
    orthcent = pca.components_[0]
#     print(pca.explained_variance_ratio_)

    baseproj = np.stack([basefeats.dot(distcent), basefeats.dot(orthcent)], axis=1)
    targproj = np.stack([targfeats.dot(distcent), targfeats.dot(orthcent)], axis=1)


    plt.plot(*baseproj.T, '.g', alpha=.03, markeredgewidth=0)
    plt.plot(*targproj.T, '.b', alpha=.03, markeredgewidth=0)

    poisonfeats = left_ops[tags == str('poison')]
    poisoncent = np.mean(poisonfeats, axis=0)
#     print("Printing below distance between centroids")
#     print(np.linalg.norm(basecent-targcent),np.linalg.norm(basecent-poisoncent), np.linalg.norm(poisoncent-targcent))
    poisonproj = np.stack([poisonfeats.dot(distcent), poisonfeats.dot(orthcent)], axis=1)
    plt.plot(*poisonproj.T, 'or', alpha=1, markeredgewidth=0, markersize=7, label='poisons')

    targetfeats = left_ops[tags == str('target')]
    targetproj = np.stack([targetfeats.dot(distcent), targetfeats.dot(orthcent)], axis=1)
    plt.plot(*targetproj.T, '^b', markersize=12, markeredgewidth=0, label='target')

#     plt.xlim(-6, 6)
    # plt.ylim(-4, 52)
    plt.xlabel('distance along centroids')
    plt.ylabel('dist along orthonormal')
    plt.legend(frameon=False, loc='lower left')
    plt.title(title)
#     plt.text(-5, 5, 'target class')
#     plt.text(2,5, 'base class')
    plt.show()
    
def generate_plot_pca(feat_path,model_path, target_class,base_class, poison_ids, title, device):

    [ops_all, labels_all, ids_all] = pickle.load( open( feat_path, "rb" ) ) 
    
    
    left_labels = np.array(labels_all)[np.in1d(labels_all,[target_class,base_class])]
    left_ids = np.array(ids_all)[np.in1d(labels_all,[target_class,base_class])]
    left_ops = ops_all[np.in1d(labels_all,[target_class,base_class])]
    id2label = dict(zip(left_ids, left_labels))

    tags = []
    for i in left_ids:
        if str(i) in poison_ids:
            tags.append('poison')
        elif i == 'target':
            tags.append('target')
        elif id2label[i]== target_class:
            tags.append(str(target_class))
        else:
            tags.append(str(base_class))

    tags = np.array(tags)  
    print(np.sum(tags == str(target_class)), 
          np.sum(tags == str(base_class)), np.sum(tags == str('poison')), np.sum(tags == str('target')))
    
    basefeats = left_ops[tags == str(base_class)]
    targfeats = left_ops[tags == str(target_class)]
    
    a = np.concatenate([basefeats, targfeats])
    pca = PCA(n_components=2)
    pca.fit(np.concatenate([basefeats, targfeats]))
    distcent = pca.components_[0]
    orthcent = pca.components_[1]
#     print(pca.explained_variance_ratio_)

    baseproj = np.stack([basefeats.dot(distcent), basefeats.dot(orthcent)], axis=1)
    targproj = np.stack([targfeats.dot(distcent), targfeats.dot(orthcent)], axis=1)


    plt.plot(*baseproj.T, '.g', alpha=.03, markeredgewidth=0)
    plt.plot(*targproj.T, '.b', alpha=.03, markeredgewidth=0)

    poisonfeats = left_ops[tags == str('poison')]
    poisoncent = np.mean(poisonfeats, axis=0)
#     print("Printing below distance between centroids")
#     print(np.linalg.norm(basecent-targcent),np.linalg.norm(basecent-poisoncent), np.linalg.norm(poisoncent-targcent))
    poisonproj = np.stack([poisonfeats.dot(distcent), poisonfeats.dot(orthcent)], axis=1)
    plt.plot(*poisonproj.T, 'or', alpha=1, markeredgewidth=0, markersize=7, label='poisons')

    targetfeats = left_ops[tags == str('target')]
    targetproj = np.stack([targetfeats.dot(distcent), targetfeats.dot(orthcent)], axis=1)
    plt.plot(*targetproj.T, '^b', markersize=12, markeredgewidth=0, label='target')

#     plt.xlim(-6, 6)
    # plt.ylim(-4, 52)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(frameon=False, loc='lower left')
    plt.title(title)
    plt.text(-5, 5, 'target class')
    plt.text(2,5, 'base class')
    plt.show()
    

    
# def bypass_last_layer(model):
#     """Hacky way of separating features and classification head for many models.
#     Patch this function if problems appear.
#     """
#     layer_cake = list(model.children())
#     last_layer = layer_cake[-1]
#     headless_model = torch.nn.Sequential(*(layer_cake[:-1]), torch.nn.Flatten())  # this works most of the time all of the time :<
#     return headless_model, last_layer
    
def bypass_last_layer(model):
    """Hacky way of separating features and classification head for many models.
    Patch this function if problems appear.

    Modified to handle Sequential containing Linear layers.
    """
    layer_cake = list(model.children())
    
    # Iterate backwards to find the last layer with weights (Linear or Conv)
    last_layer = None
    headless_layers = []
    found_last = False

    for layer in reversed(layer_cake):
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            if not found_last:
                last_layer = layer
                found_last = True
            else:
                headless_layers.insert(0, layer)
        elif isinstance(layer, nn.Sequential):
            # If it's a sequential, look inside for the last linear/conv layer
            seq_children = list(layer.children())
            found_in_seq = False
            temp_seq_layers = []
            for sub_layer in reversed(seq_children):
                if isinstance(sub_layer, (nn.Linear, nn.Conv2d)):
                    if not found_last:
                        last_layer = sub_layer
                        found_last = True
                        found_in_seq = True
                    else:
                        temp_seq_layers.insert(0, sub_layer)
                else:
                     temp_seq_layers.insert(0, sub_layer)
            
            if temp_seq_layers:
                 headless_layers.insert(0, nn.Sequential(*temp_seq_layers))
        else:
             headless_layers.insert(0, layer)

    if last_layer is None:
        raise ValueError("Could not find a layer with weights in the model.")

    headless_model = torch.nn.Sequential(*headless_layers, torch.nn.Flatten())

    return headless_model, last_layer

    
def generate_plot_centroid_3d_labels(feat_path, model_path, target_class,base_class, poison_ids, title, device):

    [ops_all, labels_all, ids_all] = pickle.load( open( feat_path, "rb" ) ) 
    # model = resnet_picker('ResNet18', 'CIFAR10')
    model = HG(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device)
    headless_model, last_layer  = bypass_last_layer(model)
    last_layer_weights = last_layer.weight.detach().cpu().numpy()
    logit_matrix = np.matmul(ops_all, last_layer_weights.T)
    classif = np.argmax(logit_matrix, axis=1)
    
    left_labels = np.array(labels_all)[np.in1d(labels_all,[target_class,base_class])]
    left_ids = np.array(ids_all)[np.in1d(labels_all,[target_class,base_class])]
    left_ops = ops_all[np.in1d(labels_all,[target_class,base_class])]
    classif = classif[np.in1d(labels_all,[target_class,base_class])]
    id2label = dict(zip(left_ids, left_labels))

    tags = []
    for i in left_ids:
        if str(i) in poison_ids:
            tags.append('poison')
        elif i == 'target':
            tags.append('target')
        elif id2label[i]== target_class:
            tags.append(str(target_class))
        else:
            tags.append(str(base_class))

    tags = np.array(tags)  
    print(np.sum(tags == str(target_class)), 
          np.sum(tags == str(base_class)), np.sum(tags == str('poison')), np.sum(tags == str('target')))
    
    basefeats = left_ops[tags == str(base_class)]
    targfeats = left_ops[tags == str(target_class)]
    basecent = np.mean(basefeats, axis=0)
    targcent = np.mean(targfeats, axis=0)
    distcent = targcent - basecent
    distcent /= np.linalg.norm(distcent)
    distcent *= -1

    basefeats_ = basefeats.dot(distcent)
    basefeats_ = np.outer(basefeats_, distcent)
    basefeats_ = basefeats - basefeats_

    targfeats_ = targfeats.dot(distcent)
    targfeats_ = np.outer(targfeats_, distcent)
    targfeats_ = targfeats - targfeats_

    a = np.concatenate([basefeats_, targfeats_])
    pca = PCA(n_components=3)
    pca.fit(np.concatenate([basefeats_, targfeats_]))
    orthcent = pca.components_[0]
#     print(pca.explained_variance_ratio_)


    allproj = np.stack([left_ops.dot(distcent), left_ops.dot(orthcent)], axis=1)
    
    data_3ax = np.column_stack((allproj, classif))
    from mpl_toolkits.mplot3d import Axes3D
#     fig = plt.figure()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    uniq_tags = np.unique(tags)
    colors = ['blue', 'green','red', 'black']
    markers = [',',',','o','^']
    alphas = [0.03,0.03,0.2,1]
    sizes = [5,5,10,50]
    col_scheme = dict(zip(uniq_tags, zip(colors, markers,alphas,sizes)))
#     print(col_scheme)
    for i in range(len(tags)):
        tag = tags[i]
        c,m,al,ms = col_scheme[tag]
        ax.scatter(data_3ax[i,0], data_3ax[i,1], data_3ax[i,2], c=c, marker=m,
                  alpha=al,s=ms)
        
    ax.set_xlabel('distance along centroids')
    ax.set_ylabel('dist along orthonormal')
    ax.set_zlabel('Predicted class')
#     handles, labels = ax.legend_elements(prop="colors", alpha=0.6)
#     legend2 = ax.legend(handles, labels, loc="upper right", title="colors")

#     plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title(title)
    plt.show()
    
def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p    
    
def generate_plot_lda_patch(feat_path,model_path, target_class,base_class, poison_ids, title, device):
    
    [ops_all, labels_all, ids_all] = pickle.load( open( feat_path, "rb" ) ) 
    left_labels = np.array(labels_all)[np.in1d(labels_all,[target_class,base_class])]
    left_ids = np.array(ids_all)[np.in1d(labels_all,[target_class,base_class])]
    left_ops = ops_all[np.in1d(labels_all,[target_class,base_class])]
    id2label = dict(zip(left_ids, left_labels))

    tags = []
    for i in left_ids:
        if str(i) in poison_ids:
            tags.append('poison')
        elif i == 'target':
            tags.append('target')
        elif id2label[i]== target_class:
            tags.append(str(target_class))
        else:
            tags.append(str(base_class))

    tags = np.array(tags)  
#     print(np.sum(tags == str(target_class)), 
#           np.sum(tags == str(base_class)), np.sum(tags == str('poison')), np.sum(tags == str('target')))

    basefeats = left_ops[tags == str(base_class)]
    targfeats = left_ops[tags == str(target_class)]
    poisonfeats = left_ops[tags == 'poison']
    targetfeats = left_ops[tags == 'target']
    basecent = np.mean(basefeats, axis=0)
    targcent = np.mean(targfeats, axis=0)
    poisoncent = np.mean(poisonfeats, axis=0)
#     print("Printing below distance between centroids")
#     print(np.linalg.norm(basecent-targcent),np.linalg.norm(basecent-poisoncent), np.linalg.norm(poisoncent-targcent))
    
    ol_tags = np.concatenate([tags[tags == str(base_class)], tags[tags == str(target_class)], tags[tags == str('poison')]])
    ol_feats = np.concatenate([basefeats, targfeats, poisonfeats])
#     print(ol_tags.shape, ol_feats.shape)
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_r2 = lda.fit(ol_feats, ol_tags).transform(ol_feats)
    
    colors = ['green', 'blue','red']
    target_names = [str(base_class), str(target_class), 'poison']
    alphas = [0.05,0.05, 0.05]
    sizes = [5,5,5]
    plt.figure()
    for color, i, target_name,al,si in zip(colors, target_names, target_names,alphas,sizes):
        plt.scatter(X_r2[ol_tags == i, 0], X_r2[ol_tags == i, 1], alpha= al, color=color,
                    label=i,s=si)
    target_proj = lda.fit(ol_feats, ol_tags).transform(targetfeats)
    plt.scatter(target_proj[:,0], target_proj[:,1], alpha=0.4, color='black', marker='^',
                    label='target',s=5)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title(title)
    plt.show()
    
def genplot_centroid_prob_2d_patch(feat_path, model_path, target_class,base_class, poison_ids, title, device):

    [ops_all, labels_all, ids_all] = pickle.load( open( feat_path, "rb" ) ) 
    # model = resnet_picker('ResNet18', 'CIFAR10')
    model = HG(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device)
    headless_model, last_layer  = bypass_last_layer(model)
    last_layer_weights = last_layer.weight.detach().cpu().numpy()
    logit_matrix = np.matmul(ops_all, last_layer_weights.T)
    softmax_scores = softmax(logit_matrix, theta = 1, axis = 1)
    target_lab_confidence = softmax_scores[:,base_class]
    
    left_labels = np.array(labels_all)[np.in1d(labels_all,[target_class,base_class])]
    left_ids = np.array(ids_all)[np.in1d(labels_all,[target_class,base_class])]
    left_ops = ops_all[np.in1d(labels_all,[target_class,base_class])]
    target_lab_confidence = target_lab_confidence[np.in1d(labels_all,[target_class,base_class])]
    id2label = dict(zip(left_ids, left_labels))

    tags = []
    for i in left_ids:
        if str(i) in poison_ids:
            tags.append('poison')
        elif i == 'target':
            tags.append('target')
        elif id2label[i]== target_class:
            tags.append('target_class')
        else:
            tags.append('poison_class')

    tags = np.array(tags)  
#     print(np.sum(tags == str(target_class)), 
#           np.sum(tags == str(base_class)), np.sum(tags == str('poison')), np.sum(tags == str('target')))
    
    basefeats = left_ops[tags == str('poison_class')]
    targfeats = left_ops[tags == str('target_class')]
    basecent = np.mean(basefeats, axis=0)
    targcent = np.mean(targfeats, axis=0)
    distcent = targcent - basecent
    distcent /= np.linalg.norm(distcent)
    distcent *= -1

    basefeats_ = basefeats.dot(distcent)
    basefeats_ = np.outer(basefeats_, distcent)
    basefeats_ = basefeats - basefeats_

    targfeats_ = targfeats.dot(distcent)
    targfeats_ = np.outer(targfeats_, distcent)
    targfeats_ = targfeats - targfeats_

    a = np.concatenate([basefeats_, targfeats_])
    pca = PCA(n_components=3)
    pca.fit(np.concatenate([basefeats_, targfeats_]))
    orthcent = pca.components_[0]


    allproj = np.stack([left_ops.dot(distcent), left_ops.dot(orthcent)], axis=1)
    
    data_3ax = np.column_stack((allproj, target_lab_confidence*30+10))
    plt.figure()
    uniq_tags = np.array(['target_class','poison_class','poison','target'])#np.unique(tags)
    colors = ['royalblue', 'green','red', 'black']
    markers = [',',',','o','^']
    alphas = [0.05,0.05,0.05,0.4]
    sizes = [5,5,5,5]
    for color, i, target_name,al,m in zip(colors, uniq_tags, uniq_tags,alphas,markers):
        s = data_3ax[tags == i, 2]/3
        if i == 'target':
            s = s*2
        else:
            s = 10
        plt.scatter(data_3ax[tags == i, 0], data_3ax[tags == i, 1], alpha= al, color=color,
                    label=i,s= s,marker=m)
    plt.xlabel('distance along centroids',fontsize=15,fontweight='medium',fontvariant='small-caps')
    plt.ylabel('distance orthonormal',fontsize=15,fontweight='medium',fontvariant='small-caps')
    figname = title.replace(" ", "_")+ "_2d.pdf"
    plt.savefig(os.path.join('./plots/2d', figname), bbox_inches='tight')
    #     plt.legend(loc='best', shadow=False, scatterpoints=1)
#     plt.title(title)
    plt.show()
    
def genplot_centroid_3d_patch(feat_path, model_path, target_class,base_class, poison_ids, title, device):

    [ops_all, labels_all, ids_all] = pickle.load( open( feat_path, "rb" ) ) 
    # model = resnet_picker('ResNet18', 'CIFAR10')
    model = HG(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device)
    headless_model, last_layer  = bypass_last_layer(model)
    last_layer_weights = last_layer.weight.detach().cpu().numpy()
    logit_matrix = np.matmul(ops_all, last_layer_weights.T)
    softmax_scores = softmax(logit_matrix, theta = 1, axis = 1)
    target_lab_confidence = softmax_scores[:,base_class]
    
    left_labels = np.array(labels_all)[np.in1d(labels_all,[target_class,base_class])]
    left_ids = np.array(ids_all)[np.in1d(labels_all,[target_class,base_class])]
    left_ops = ops_all[np.in1d(labels_all,[target_class,base_class])]
    target_lab_confidence = target_lab_confidence[np.in1d(labels_all,[target_class,base_class])]
    id2label = dict(zip(left_ids, left_labels))

    tags = []
    for i in left_ids:
        if str(i) in poison_ids:
            tags.append('poison')
        elif i == 'target':
            tags.append('target')
        elif id2label[i]== target_class:
            tags.append(str('target_class'))
        else:
            tags.append(str('poison_class'))

    tags = np.array(tags)  
#     print(np.sum(tags == str(target_class)), 
#           np.sum(tags == str(base_class)), np.sum(tags == str('poison')), np.sum(tags == str('target')))
    
    basefeats = left_ops[tags == str('poison_class')]
    targfeats = left_ops[tags == str('target_class')]
    basecent = np.mean(basefeats, axis=0)
    targcent = np.mean(targfeats, axis=0)
    distcent = targcent - basecent
    distcent /= np.linalg.norm(distcent)
    distcent *= -1

    basefeats_ = basefeats.dot(distcent)
    basefeats_ = np.outer(basefeats_, distcent)
    basefeats_ = basefeats - basefeats_

    targfeats_ = targfeats.dot(distcent)
    targfeats_ = np.outer(targfeats_, distcent)
    targfeats_ = targfeats - targfeats_

    a = np.concatenate([basefeats_, targfeats_])
    pca = PCA(n_components=3)
    pca.fit(np.concatenate([basefeats_, targfeats_]))
    orthcent = pca.components_[0]

    allproj = np.stack([left_ops.dot(distcent), left_ops.dot(orthcent)], axis=1)
    
    data_3ax = np.column_stack((allproj, target_lab_confidence))
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    uniq_tags = np.array(['target_class','poison_class','poison','target'])
    colors = ['royalblue', 'green','red', 'dimgray']
    markers = [',',',','o','^']
    alphas = [0.03,0.03,0.05,0.2]
    sizes = [5,5,5,5]
    for color, i, target_name,al,si,m in zip(colors, uniq_tags, uniq_tags,alphas,sizes,markers):
        
        ax.scatter(data_3ax[tags == i, 0], data_3ax[tags == i, 1], 
                   data_3ax[tags == i, 2],alpha= al, color=color,
                     marker=m, label=i,s=si)
        
    ax.set_xlabel('distance along centroids',fontsize=15,fontweight='bold',fontvariant='small-caps')
    ax.set_ylabel('distance orthonormal',fontsize=15,fontweight='bold',fontvariant='small-caps')
    ax.set_zlabel('poison class probability',fontsize=15,fontweight='bold',fontvariant='small-caps')
    figname = title.replace(" ", "_")+ "_3d.pdf"
    plt.savefig(os.path.join('./plots', figname), bbox_inches='tight')
#     plt.title(title)
    plt.show()
    
def generate_plot_lda(feat_path,model_path, target_class,base_class, poison_ids, title, device):
    
    [ops_all, labels_all, ids_all] = pickle.load( open( feat_path, "rb" ) ) 
    left_labels = np.array(labels_all)[np.in1d(labels_all,[target_class,base_class])]
    left_ids = np.array(ids_all)[np.in1d(labels_all,[target_class,base_class])]
    left_ops = ops_all[np.in1d(labels_all,[target_class,base_class])]
    id2label = dict(zip(left_ids, left_labels))

    tags = []
    for i in left_ids:
        if str(i) in poison_ids:
            tags.append('poison')
        elif i == 'target':
            tags.append('target')
        elif id2label[i]== target_class:
            tags.append(str(target_class))
        else:
            tags.append(str(base_class))

    tags = np.array(tags)  

    basefeats = left_ops[tags == str(base_class)]
    targfeats = left_ops[tags == str(target_class)]
    poisonfeats = left_ops[tags == 'poison']
    targetfeats = left_ops[tags == 'target']
    basecent = np.mean(basefeats, axis=0)
    targcent = np.mean(targfeats, axis=0)
    poisoncent = np.mean(poisonfeats, axis=0)
   
    ol_tags = np.concatenate([tags[tags == str(base_class)], tags[tags == str(target_class)], tags[tags == str('poison')]])
    ol_feats = np.concatenate([basefeats, targfeats, poisonfeats])
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_r2 = lda.fit(ol_feats, ol_tags).transform(ol_feats)
    
    colors = ['green', 'royalblue','red']
    target_names = [str(base_class), str(target_class), 'poison']
    alphas = [0.02,0.02,0.5]
    sizes = [5,5,10]
    plt.figure()
    for color, i, target_name,al,si in zip(colors, target_names, target_names,alphas,sizes):
        plt.scatter(X_r2[ol_tags == i, 0], X_r2[ol_tags == i, 1], alpha= al, color=color,
                    label=i,s=si)
    target_proj = lda.fit(ol_feats, ol_tags).transform(targetfeats)
    plt.scatter(target_proj[0][0], target_proj[0][1], alpha=1, color='dimgray', marker='^',label='target',s=50,edgecolors = 'black', linewidth=3)
    plt.xlabel('LD1',fontsize=15,fontweight='medium',fontvariant='small-caps')
    plt.ylabel('LD2',fontsize=15,fontweight='medium',fontvariant='small-caps')
    figname = title.replace(" ", "_")+ "_3d_lda.pdf"
    plt.savefig(os.path.join('./plots', figname), bbox_inches='tight')
    #     plt.legend(loc='best', shadow=False, scatterpoints=1)
#     plt.title(title)
    plt.show()

def genplot_centroid_prob_3d(feat_path, model_path, target_class,base_class, poison_ids, title, device):

    ## [ops_all, labels_all, ids_all] = pickle.load( open( feat_path, "rb" ) )
    ops_all, labels_all, ids_all = pickle.load(open(feat_path, "rb"))

    # # sketchy
    # ops_all = ops_all[:, :128]
    # # labels_all = labels_all[:, :128]
    # labels_all = np.array(labels_all)
    # # ids_all = ids_all[:, :128]
    # ids_all = np.array(ids_all)

    print("model path:", model_path)
    print("feat_path:", feat_path)

    # model = resnet_picker('ResNet18', 'CIFAR10')
    model = HG(num_classes=NUM_CLASSES) 
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device)
    headless_model, last_layer  = bypass_last_layer(model)
    last_layer_weights = last_layer.weight.detach().cpu().numpy()

    # Get the intermediate projection block: Linear(256→128) + ReLU + Dropout
    projection_block = torch.nn.Sequential(*list(model.fc.children())[:3])
    projection_block.to(device)
    projection_block.eval()

    # Project the 256-dim features to 128-dim
    with torch.no_grad():
        features_tensor = torch.tensor(ops_all, dtype=torch.float32).to(device)
        projected_features = projection_block(features_tensor).cpu().numpy()

    print("ops_all shape:", ops_all.shape)
    print("projected features shape:", projected_features.shape)
    print("last_layer_weights shape:", last_layer_weights.shape)

    logit_matrix = np.matmul(projected_features, last_layer_weights.T)
    softmax_scores = softmax(logit_matrix, theta = 1, axis = 1)
    target_lab_confidence = softmax_scores[:,base_class]
    
    left_labels = np.array(labels_all)[np.in1d(labels_all,[target_class,base_class])]
    left_ids = np.array(ids_all)[np.in1d(labels_all,[target_class,base_class])]
    left_ops = ops_all[np.in1d(labels_all,[target_class,base_class])]
    target_lab_confidence = target_lab_confidence[np.in1d(labels_all,[target_class,base_class])]
    id2label = dict(zip(left_ids, left_labels))

    tags = []
    for i in left_ids:
        if str(i) in poison_ids:
            tags.append('poison')
        elif i == 'target':
            tags.append('target')
        elif id2label[i]== target_class:
            tags.append('target_class')
        else:
            tags.append('poison_class')

    tags = np.array(tags)  
#     print(np.sum(tags == str(target_class)), 
#           np.sum(tags == str(base_class)), np.sum(tags == str('poison')), np.sum(tags == str('target')))
    
    basefeats = left_ops[tags == str('poison_class')]
    targfeats = left_ops[tags == str('target_class')]
    basecent = np.mean(basefeats, axis=0)
    targcent = np.mean(targfeats, axis=0)
    distcent = targcent - basecent
    distcent /= np.linalg.norm(distcent)
    distcent *= -1

    basefeats_ = basefeats.dot(distcent)
    basefeats_ = np.outer(basefeats_, distcent)
    basefeats_ = basefeats - basefeats_

    targfeats_ = targfeats.dot(distcent)
    targfeats_ = np.outer(targfeats_, distcent)
    targfeats_ = targfeats - targfeats_

    a = np.concatenate([basefeats_, targfeats_])
    pca = PCA(n_components=3)
    pca.fit(np.concatenate([basefeats_, targfeats_]))
    orthcent = pca.components_[0]


    allproj = np.stack([left_ops.dot(distcent), left_ops.dot(orthcent)], axis=1)
    
    data_3ax = np.column_stack((allproj, target_lab_confidence))
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    uniq_tags = np.array(['target_class','poison_class','poison','target'])#np.unique(tags)
    colors = ['royalblue', 'green','red', 'dimgray']
    markers = [',',',','o','^']
    alphas = [0.03,0.03,0.3,1]
    sizes = [5,5,10,200]
    for color, i, target_name,al,si,m in zip(colors, uniq_tags, uniq_tags,alphas,sizes,markers):
        if i == 'target':
            edgecolors = 'black'
            linewidth=3
        else:
            edgecolors= None
            linewidth =0
        ax.scatter(data_3ax[tags == i, 0], data_3ax[tags == i, 1], 
                   data_3ax[tags == i, 2],alpha= al, color=color,
                     marker=m, label=i,s=si,
                   edgecolors=edgecolors,linewidth=linewidth)
        
    ax.set_xlabel('distance along centroids',fontsize=15,fontweight='bold',fontvariant='small-caps')
    ax.set_ylabel('distance orthonormal',fontsize=15,fontweight='bold',fontvariant='small-caps')
    ax.set_zlabel('poison class probability',fontsize=15,fontweight='bold',fontvariant='small-caps')
    figname = title.replace(" ", "_")+ "_3d.pdf"
    plt.savefig(os.path.join('./plots/3d', figname), bbox_inches='tight')
#     plt.title(title)
    plt.show()
    
    
def genplot_centroid_prob_2d(feat_path, model_path, target_class,base_class, poison_ids, title, device):

    # [ops_all, labels_all, ids_all] = pickle.load( open( feat_path, "rb" ) )
    ops_all, labels_all, ids_all = pickle.load(open(feat_path, "rb"))
    # # sketchy
    # ops_all = ops_all[:, :128]
    # # labels_all = labels_all[:, :128]
    # labels_all = np.array(labels_all)
    # # ids_all = ids_all[:, :128]
    # ids_all = np.array(ids_all)

    print("model path:", model_path)
    print("feat_path:", feat_path)
    # model = resnet_picker('ResNet18', 'CIFAR10')
    model = HG(num_classes=NUM_CLASSES) 
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device)
    headless_model, last_layer  = bypass_last_layer(model)
    last_layer_weights = last_layer.weight.detach().cpu().numpy()

    # Get the intermediate projection block: Linear(256→128) + ReLU + Dropout
    projection_block = torch.nn.Sequential(*list(model.fc.children())[:3])
    projection_block.to(device)
    projection_block.eval()

    # Project the 256-dim features to 128-dim
    with torch.no_grad():
        features_tensor = torch.tensor(ops_all, dtype=torch.float32).to(device)
        projected_features = projection_block(features_tensor).cpu().numpy()

    print("ops_all shape:", ops_all.shape)
    print("projected features shape:", projected_features.shape)
    print("last_layer_weights shape:", last_layer_weights.shape)

    logit_matrix = np.matmul(projected_features, last_layer_weights.T)
    softmax_scores = softmax(logit_matrix, theta = 1, axis = 1)
    target_lab_confidence = softmax_scores[:,base_class]
    
    left_labels = np.array(labels_all)[np.in1d(labels_all,[target_class,base_class])]
    left_ids = np.array(ids_all)[np.in1d(labels_all,[target_class,base_class])]
    left_ops = ops_all[np.in1d(labels_all,[target_class,base_class])]
    target_lab_confidence = target_lab_confidence[np.in1d(labels_all,[target_class,base_class])]
    id2label = dict(zip(left_ids, left_labels))

    tags = []
    for i in left_ids:
        if str(i) in poison_ids:
            tags.append('poison')
        elif i == 'target':
            tags.append('target')
        elif id2label[i]== target_class:
            tags.append('target_class')
        else:
            tags.append('poison_class')

    tags = np.array(tags)  
    print(np.sum(tags == str(target_class)), 
          np.sum(tags == str(base_class)), np.sum(tags == str('poison')), np.sum(tags == str('target')))
    
    basefeats = left_ops[tags == str('poison_class')]
    targfeats = left_ops[tags == str('target_class')]
    basecent = np.mean(basefeats, axis=0)
    targcent = np.mean(targfeats, axis=0)
    distcent = targcent - basecent
    distcent /= np.linalg.norm(distcent)
    distcent *= -1

    basefeats_ = basefeats.dot(distcent)
    basefeats_ = np.outer(basefeats_, distcent)
    basefeats_ = basefeats - basefeats_

    targfeats_ = targfeats.dot(distcent)
    targfeats_ = np.outer(targfeats_, distcent)
    targfeats_ = targfeats - targfeats_

    a = np.concatenate([basefeats_, targfeats_])
    pca = PCA(n_components=3)
    pca.fit(np.concatenate([basefeats_, targfeats_]))
    orthcent = pca.components_[0]


    allproj = np.stack([left_ops.dot(distcent), left_ops.dot(orthcent)], axis=1)
    
    data_3ax = np.column_stack((allproj, target_lab_confidence*30+10))
    plt.figure()
    uniq_tags = np.array(['target_class','poison_class','poison','target'])#np.unique(tags)
    colors = ['royalblue', 'green','red', 'black']
    markers = [',',',','o','^']
    alphas = [0.01,0.01,0.08,1]
    sizes = [5,5,10,200]
    for color, i, target_name,al,m in zip(colors, uniq_tags, uniq_tags,alphas,markers):
        s = data_3ax[tags == i, 2]
        if i == 'target':
            s = s*10
        else:
            s = 10
        plt.scatter(data_3ax[tags == i, 0], data_3ax[tags == i, 1], alpha= al, color=color,
                    label=i,s= s,marker=m)
    plt.xlabel('distance along centroids',fontsize=15,fontweight='medium',fontvariant='small-caps')
    plt.ylabel('distance orthonormal',fontsize=15,fontweight='medium',fontvariant='small-caps')
    figname = title.replace(" ", "_")+ "_2d.pdf"
    plt.savefig(os.path.join('./plots/2d', figname), bbox_inches='tight')
    #     plt.legend(loc='best', shadow=False, scatterpoints=1)
#     plt.title(title)
    plt.show()

def plot_feature_pca_all_classes(feat_path, base_class, target_class, class_names=None, dim=2, title=None, save_path=None):
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

def generate_plots(main_path, model_name, plot_function, target_class, base_class, poison_ids, device):
    os.makedirs('./plots/2d', exist_ok=True)
    os.makedirs('./plots/3d', exist_ok=True)
    
    # Plot clean model
    feat_path = os.path.join(main_path+'clean_model', 'clean_features.pickle')
    model_path = os.path.join(main_path+'clean_model', 'clean.pth')
    plot_function(feat_path, model_path, target_class, base_class, poison_ids,
                model_name + " (clean)", device)
    
    # Plot poisoned model
    feat_path = os.path.join(main_path+'defended_model', 'def_features.pickle') 
    model_path = os.path.join(main_path+'defended_model', 'def.pth')
    plot_function(feat_path, model_path, target_class, base_class, poison_ids,
                model_name+ " (poisoned)", device)

# def generate_all_embeddings_plots(main_path, model_name, base_class, target_class, save_path):  
#     os.makedirs(save_path, exist_ok=True)

#     for dim in [2,3]:
#         plot_feature_pca_all_classes(
#             feat_path=os.path.join(main_path +'clean_model', 'clean_features.pickle'),  
#             base_class,
#             target_class,
#             class_names=eurosat_classes,
#             dim=dim,
#             title=f"{model_name} PCA (2D) - Clean",
#             save_path=os.path.join(save_path, f'pca_clean_{dim}d.pdf')
#         )
#         plot_feature_pca_all_classes(
#             feat_path=os.path.join(main_path +'defended_model', 'def_features.pickle'), 
#             base_class, 
#             target_class,
#             class_names=eurosat_classes,
#             dim=dim,
#             title=f"{model_name} PCA (2D) - Poisoned",
#             save_path=os.path.join(save_path, f"pca_poisoned_{dim}d.pdf")
#         )
def generate_all_embeddings_plots(main_path, model_name, base_class, target_class, save_path):
    """Generate 2D and 3D PCA plots for both clean and poisoned models."""
    os.makedirs(save_path, exist_ok=True)

    # Validate paths first
    clean_feat_path = os.path.join(main_path+'clean_model', 'clean_features.pickle')
    poisoned_feat_path = os.path.join(main_path+'defended_model', 'def_features.pickle')
    
    if not all(os.path.exists(p) for p in [clean_feat_path, poisoned_feat_path]):
        missing = [p for p in [clean_feat_path, poisoned_feat_path] if not os.path.exists(p)]
        raise FileNotFoundError(f"Missing feature files: {missing}")

    for dim in [2, 3]:
        # Clean model plot
        plot_feature_pca_all_classes(
            feat_path=clean_feat_path,
            base_class=base_class,
            target_class=target_class,
            class_names=eurosat_classes,
            dim=dim,
            title=f"{model_name} PCA ({dim}D) - Clean",
            save_path=os.path.join(save_path, f'pca_clean_{dim}d.pdf')
        )

        # Poisoned model plot
        plot_feature_pca_all_classes(
            feat_path=poisoned_feat_path,
            base_class=base_class,
            target_class=target_class,
            class_names=eurosat_classes,
            dim=dim,
            title=f"{model_name} PCA ({dim}D) - Poisoned",
            save_path=os.path.join(save_path, f'pca_poisoned_{dim}d.pdf')
        )    
