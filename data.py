import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_transform(transform_list):
    """
    Build an Albumentations Compose transform from a list of op specs.

    Args:
        transform_list (list): A list of dicts, each with a single key (Albumentations class name)
                               and its parameters.
    Returns:
        A.Compose object of the specified transforms.
    """
    ops = []
    for op in transform_list:
        for name, params in op.items():
            cls = getattr(A, name)
            # Some ops like ToTensorV2 have no parameters
            if params is None or params == {}:
                ops.append(cls())
            else:
                ops.append(cls(**params))
    return A.Compose(ops)


class ClassAwareImageFolder(ImageFolder):
    """
    ImageFolder that applies class-specific transforms based on target index.
    """
    def __init__(self, root, transform_dict, default_transform=None):
        super().__init__(root)
        self.transform_dict = transform_dict
        self.default_transform = default_transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = np.array(self.loader(path))
        transform = self.transform_dict.get(target, self.default_transform)
        if transform:
            image = transform(image=image)["image"]
        return image, target


class AlbumentationsImageFolder(ImageFolder):
    """
    ImageFolder that applies a single Albumentations transform to all samples.
    """
    def __init__(self, root, transform):
        super().__init__(root)
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = np.array(self.loader(path))
        image = self.transform(image=image)["image"]
        return image, target


def load_data(config):
    """
    Load training and validation DataLoaders based on the provided config.

    Args:
        config (dict): Parsed YAML config with data and augmentation settings.

    Returns:
        train_loader, val_loader, classes
    """
    # Build transforms from config
    aug_cfg = config['augmentation']
    transform_A    = build_transform(aug_cfg['transform_A'])
    transform_B    = build_transform(aug_cfg['transform_B'])
    transform_C    = build_transform(aug_cfg['transform_C'])
    transform_test = build_transform(aug_cfg['transform_test'])

    transform_dict = {0: transform_A, 1: transform_B, 2: transform_C}

    # Load dataset without transforms to get class names and split
    base_dataset = ImageFolder(root=config['data']['train_dir'])
    classes = base_dataset.classes
    n = len(base_dataset)
    val_size = int(config['data']['val_split'] * n)
    train_size = n - val_size

    # Prepare class-aware and test datasets
    train_dataset = ClassAwareImageFolder(
        root=config['data']['train_dir'],
        transform_dict=transform_dict,
        default_transform=transform_C
    )
    val_dataset = AlbumentationsImageFolder(
        root=config['data']['train_dir'],
        transform=transform_test
    )

    # Create train/val splits
    indices = torch.randperm(n).tolist()
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]
    train_subset = torch.utils.data.Subset(train_dataset, train_idx)
    val_subset = torch.utils.data.Subset(val_dataset, val_idx)

    # DataLoader parameters
    batch_size = config['training']['batch_size']
    num_workers = config['training'].get('num_workers', 4)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, classes