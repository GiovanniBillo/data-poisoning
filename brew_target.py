"""General interface script to launch poisoning jobs.

Run this script from the top folder."""

import torch

import datetime
import time
import os
import numpy as np
import pickle
from PIL import Image
import torchvision.transforms as T
# import random # REMOVED: Not needed as we are not saving random train images

import forest
from forest.filtering_defenses import get_defense

torch.backends.cudnn.benchmark = forest.consts.BENCHMARK
torch.multiprocessing.set_sharing_strategy(forest.consts.SHARING_STRATEGY)

# Place this function definition near the top, e.g., after imports
def unnormalize_img(img_tensor, mean, std):
    """
    Unnormalizes a tensor image.
    Args:
        img_tensor (torch.Tensor): The normalized image tensor (C, H, W).
        mean (list or tuple): The mean values used for normalization (per channel).
        std (list or tuple): The std values used for normalization (per channel).
    Returns:
        torch.Tensor: The unnormalized image tensor.
    """
    # Ensure mean/std are on the same device as img_tensor for element-wise ops
    mean = torch.tensor(mean).view(-1, 1, 1).to(img_tensor.device)
    std = torch.tensor(std).view(-1, 1, 1).to(img_tensor.device)
    unnormalized_tensor = img_tensor * std + mean
    return unnormalized_tensor

# Parse input arguments
args = forest.options().parse_args()
# 100% reproducibility?
if args.deterministic:
    forest.utils.set_deterministic()

subfolder = args.modelsave_path
clean_path = os.path.join(subfolder, f'{args.net}_{args.dataset}_{args.eps}_clean_model')
def_model_path = os.path.join(subfolder, f'{args.net}_{args.dataset}_{args.eps}_defended_model')

for char in ["[", "]", "'"]:
    clean_path = clean_path.replace(char, '').strip()
    def_model_path = def_model_path.replace(char, '').strip()

os.makedirs(clean_path, exist_ok=True)
os.makedirs(def_model_path, exist_ok=True)

def get_features(model, data, poison_delta):
    feats = np.array([])
    targets = []
    indices = []
    layer_cake = list(model.model.children())
    feature_extractor = torch.nn.Sequential(*(layer_cake[:-1]), torch.nn.Flatten())
    with torch.no_grad():
        for i, (img, target, idx) in enumerate(data.trainset):
            lookup = data.poison_lookup.get(idx)
            if lookup is not None and poison_delta is not None:
                img += poison_delta[lookup, :, :, :]
            img = img.unsqueeze(0).to(**data.setup)
            f = feature_extractor(img).detach().cpu().numpy()
            feats = np.copy(f) if i == 0 else np.append(feats, f, axis=0)
            targets.append(target)
            indices.append(idx)

        for enum, (img, target, idx) in enumerate(data.targetset):
            targets.append(target)
            indices.append('target')
            img = img.unsqueeze(0).to(**data.setup)
            f = feature_extractor(img).detach().cpu().numpy()
            feats = np.append(feats, f, axis=0)
    return feats, targets, indices

if __name__ == "__main__":

    setup = forest.utils.system_startup(args)

    model = forest.Victim(args, setup=setup)
    data = forest.Kettle(args, model.defs.batch_size, model.defs.augmentations,
                         model.defs.mixing_method, setup=setup)
    witch = forest.Witch(args, setup=setup)
    witch.patch_targets(data)

    # --- Determine Dataset Mean and Std for Unnormalization ---
    # IMPORTANT: You MUST replace these with the actual mean and std
    # used by your specific dataset (e.g., CIFAR-10, ImageNet).
    # Check forest/consts.py or where your dataset's transforms are defined.
    #
    # Example for CIFAR-10:
    # DATASET_MEAN = [0.4914, 0.4822, 0.4465]
    # DATASET_STD = [0.2023, 0.1994, 0.2010]
    #
    # Example for ImageNet (common default if not specified):
    DATASET_MEAN = [0.485, 0.456, 0.406] # Placeholder, replace with actual values
    DATASET_STD = [0.229, 0.224, 0.225]   # Placeholder, replace with actual values


    # Save target image
    target_img_dir = getattr(args, 'target_save_path', './saved_target')
    os.makedirs(target_img_dir, exist_ok=True)
    if len(data.targetset) > 0:
        target_img_tensor, target_label, _ = data.targetset[0]
        to_pil = T.ToPILImage()

        # Save the original clamped version (this will likely be the distorted one if normalized)
        # Renamed to make it clear this is the clamped-only output.
        target_img_clamped = to_pil(target_img_tensor.cpu().detach().clamp(0, 1))
        target_img_clamped.save(os.path.join(target_img_dir, 'target_clamped.png'))
        print(f"[\u2713] Clamped target image saved to {os.path.join(target_img_dir, 'target_clamped.png')}")

        # Save the unnormalized version (this should look correct)
        unnormalized_target_tensor = unnormalize_img(target_img_tensor.cpu().detach(), DATASET_MEAN, DATASET_STD)
        # Clamp after unnormalization to ensure values are within [0, 1] for image saving
        unnormalized_target_tensor = unnormalized_target_tensor.clamp(0, 1)
        target_img_unnormalized = to_pil(unnormalized_target_tensor)
        target_img_unnormalized.save(os.path.join(target_img_dir, 'target_unnormalized.png'))
        print(f"[\u2713] Unnormalized target image saved to {os.path.join(target_img_dir, 'target_unnormalized.png')}")

        # Save tensor statistics for debugging the numerical range
        with open(os.path.join(target_img_dir, 'target_tensor_stats.txt'), 'w') as file:
            file.write(f"Tensor Min: {target_img_tensor.min().item()}\n")
            file.write(f"Tensor Max: {target_img_tensor.max().item()}\n")
            file.write(f"Tensor Mean: {target_img_tensor.mean().item()}\n")
            file.write(f"Tensor Std: {target_img_tensor.std().item()}\n")
            file.write(f"Tensor Shape: {target_img_tensor.shape}\n")
        print(f"[\u2713] Target image tensor stats saved to {os.path.join(target_img_dir, 'target_tensor_stats.txt')}")

    else:
        print("[!] No target image found to save.")

    start_time = time.time()
    if args.pretrained_model:
        print('Loading pretrained model...')
        stats_clean = None
    elif args.skip_clean_training:
        print('Skipping clean training...')
        stats_clean = None
    else:
        stats_clean = model.train(data, max_epoch=args.max_epoch)
    train_time = time.time()

    torch.save(model.model.state_dict(), os.path.join(clean_path, 'clean.pth'))
    model.model.eval()
    feats, targets, indices = get_features(model, data, poison_delta=None)
    with open(os.path.join(clean_path, 'clean_features.pickle'), 'wb+') as file:
        pickle.dump([feats, targets, indices], file, protocol=pickle.HIGHEST_PROTOCOL)
    model.model.train()

    poison_delta = witch.brew(model, data)
    brew_time = time.time()
    with open(os.path.join(subfolder, f'{args.net}_{args.dataset}_{args.eps}_poison_indices.pickle'), 'wb+') as file:
        pickle.dump(data.poison_ids, file, protocol=pickle.HIGHEST_PROTOCOL)
    print('Poison ids saved')

    if args.filter_defense != '':
        if args.scenario == 'from-scratch':
            model.validate(data, poison_delta)
        print('Attempting to filter poison images...')
        defense = get_defense(args)
        clean_ids = defense(data, model, poison_delta)
        poison_ids = set(range(len(data.trainset))) - set(clean_ids)
        removed_images = len(data.trainset) - len(clean_ids)
        removed_poisons = len(set(data.poison_ids.tolist()) & poison_ids)

        data.reset_trainset(clean_ids)
        print(f'Filtered {removed_images} images out of {len(data.trainset.dataset)}. {removed_poisons} were poisons.')
        filter_stats = dict(removed_poisons=removed_poisons, removed_images_total=removed_images)
    else:
        filter_stats = dict()

    if not args.pretrained_model and args.retrain_from_init:
        stats_rerun = model.retrain(data, poison_delta)
    else:
        stats_rerun = None

    torch.save(model.model.state_dict(), os.path.join(def_model_path, 'def.pth'))
    model.model.eval()
    feats, targets, indices = get_features(model, data, poison_delta=poison_delta)
    with open(os.path.join(def_model_path, 'def_features.pickle'), 'wb+') as file:
        pickle.dump([feats, targets, indices], file, protocol=pickle.HIGHEST_PROTOCOL)
    model.model.train()

    if args.vnet is not None:
        train_net = args.net
        args.net = args.vnet
        if args.vruns > 0:
            model = forest.Victim(args, setup=setup)
            stats_results = model.validate(data, poison_delta)
        else:
            stats_results = None
        args.net = train_net
    else:
        if args.vruns > 0:
            stats_results = model.validate(data, poison_delta)
        else:
            stats_results = None

    test_time = time.time()
    timestamps = dict(train_time=str(datetime.timedelta(seconds=train_time - start_time)).replace(',', ''),
                      brew_time=str(datetime.timedelta(seconds=brew_time - train_time)).replace(',', ''),
                      test_time=str(datetime.timedelta(seconds=test_time - brew_time)).replace(',', ''))

    results = (stats_clean, stats_rerun, stats_results)
    forest.utils.record_results(data, witch.stat_optimal_loss, results,
                                args, model.defs, model.model_init_seed, extra_stats={**filter_stats, **timestamps})

    if args.save is not None:
        data.export_poison(poison_delta, path=args.poison_path, mode=args.save, net=args.net, dataset=args.dataset, eps=args.eps)

    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print('---------------------------------------------------')
    print(f'Finished computations with train time: {str(datetime.timedelta(seconds=train_time - start_time))}')
    print(f'--------------------------- brew time: {str(datetime.timedelta(seconds=brew_time - train_time))}')
    print(f'--------------------------- test time: {str(datetime.timedelta(seconds=test_time - brew_time))}')
    print('-------------Job finished.-------------------------')
