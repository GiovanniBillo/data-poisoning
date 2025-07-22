from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import Dataset
import os

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),  # 50% chance of flip
    transforms.RandomAffine(
        degrees=0,  # No rotation
        shear=0.2,  # Shear range of 0.2 radians (~11.5 degrees)
        scale=(0.8, 1.2)  # Random zoom between 80%-120%
    )
    #transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
    #                    std=[0.229, 0.224, 0.225])
])

# Validation transforms (no augmentation)
test_transform = transforms.Compose([
    transforms.ToTensor()
    #transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                    std=[0.229, 0.224, 0.225])
])

def transform_batch(example, transform):
    example['image'] = transform(example['image'])
    return example

def prepare_dataset():
    ds = load_dataset("blanchon/EuroSAT_RGB")
    ds["train"] = ds["train"].map(lambda x: transform_batch(x, train_transform))
    ds["test"] = ds["test"].map(lambda x: transform_batch(x, test_transform))
    ds["train"].set_format(type='torch', columns=['image', 'label'])
    ds["test"].set_format(type='torch', columns=['image', 'label'])

    train_loader = DataLoader(ds["train"], batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(ds["test"], batch_size=BATCH_SIZE)
    num_classes = len(ds["train"].features["label"].names)
    return train_loader, test_loader, num_classes

# class EuroSATDataset(Dataset):
#     def __init__(self, hf_dataset, transform=None):
#         self.ds = hf_dataset
#         self.transform = transform
#         self.classes = hf_dataset.features["label"].names

#     def __getitem__(self, idx):
#         image = self.ds[idx]['image']
#         label = self.ds[idx]['label']
#         if self.transform:
#             image = self.transform(image)
#         return image, label

#     def get_target(self, index):
#         """Return only the target and its id.

#         Args:
#             index (int): Index

#         Returns:
#             tuple: (target, idx) where target is class_index of the target class.

#         """
#         target = self.targets[index]

#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         return target, index
#     def __len__(self):
#         return len(self.ds)
class EuroSATDataset(Dataset):
    def __init__(self, hf_dataset, transform=None, target_transform=None):
        self.ds = hf_dataset
        self.transform = transform
        self.target_transform = target_transform
        self.classes = hf_dataset.features["label"].names
        self.targets = [example["label"] for example in self.ds]  # Add this line

    def __getitem__(self, idx):
        idx = int(idx)
        image = self.ds[idx]['image']
        label = self.targets[idx]  # Use the cached targets
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, idx

    def get_target(self, index):
        target = self.targets[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return target, index

    def __len__(self):
        return len(self.ds)

def get_EUROSAT(args, test_transform=None, train=True):
    ds = load_dataset("blanchon/EuroSAT_RGB", download_mode="force_redownload")
    subfolder = os.path.join(args.modelsave_path, f'{args.net}_{args.dataset}_{args.eps}_validation_images')
    # Correzione: usa la variabile 'subfolder', non la stringa "subfolder"
    os.makedirs(subfolder, exist_ok = True)
    to_pil = transforms.ToPILImage()

    if train:
        return EuroSATDataset(ds["train"], transform=train_transform)
    else:
        # La variabile 'split' viene definita qui per essere accessibile in tutto il blocco 'else'
        split = "test" if "test" in ds else "validation"

        # Correzione: si usa '==' per il confronto, non '='
        if args.save == "only_modified":
            # Correzione: i nomi delle variabili non possono contenere '-'
            pre_transform_folder = os.path.join(subfolder,"_pre-transform")
            os.makedirs(pre_transform_folder, exist_ok = True)
            for char in ["[","]","'"]:
                pre_transform_folder = pre_transform_folder.replace(char, '')
                pre_transform_folder = pre_transform_folder.strip()


            # Itera per salvare le prime 5 immagini una per una
            for i in range(5):
                # Estrae l'immagine originale (solitamente già un oggetto PIL)
                image_pre = ds[split][i]['image']
                filename_pre = f"image_pre_{i}.png"
                image_pre.save(os.path.join(pre_transform_folder, filename_pre))

        dsv = EuroSATDataset(ds[split], transform=test_transform)

        # Correzione: si usa '==' per il confronto, non '='
        if args.save == "only_modified":
            # Correzione: i nomi delle variabili non possono contenere '-'
            post_transform_folder = os.path.join(subfolder,"_post-transform")
            os.makedirs(post_transform_folder, exist_ok = True)
            for char in ["[","]","'"]:
                post_transform_folder = post_transform_folder.replace(char, '')
                post_transform_folder = post_transform_folder.strip()

            # Itera per salvare le prime 5 immagini trasformate
            for i in range(5):
                # Estrae l'immagine trasformata (un tensore)
                tensor_img = dsv[i][0]
                image_post = to_pil(tensor_img)
                filename_post = f"image_post_{i}.png"
                image_post.save(os.path.join(post_transform_folder, filename_post))

        return dsv

# def get_EUROSAT(train=True):
#     ds = load_dataset("blanchon/EuroSAT_RGB")
#     if train:
#         ds["train"] = ds["train"].map(lambda x: transform_batch(x, train_transform))
#         ds["train"].set_format(type='torch', columns=['image', 'label'])
#     else:
#         ds["validation"] = ds["validation"].map(lambda x: transform_batch(x, test_transform))
#         ds["validation"].set_format(type='torch', columns=['image', 'label'])
#     return ds
   
