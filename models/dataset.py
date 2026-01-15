"""Wildfire dataset for PyTorch."""
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# ImageNet normalization (for pretrained weights)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Dataset mean/std for 6-band (train split only, clipped to [0,1])
SIXBAND_MEAN = [0.069138, 0.095986, 0.113498, 0.249893, 0.266409, 0.197983]  # B02,B03,B04,B08,B11,B12
SIXBAND_STD = [0.049244, 0.055162, 0.075728, 0.097269, 0.092935, 0.092973]


def get_transforms(train: bool, channels: str):
    """Get transforms with proper normalization."""
    transforms = []
    
    if train:
        transforms += [
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(15),
        ]
    
    # Normalization (critical for pretrained weights)
    if channels == "rgb":
        transforms.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    else:
        transforms.append(T.Normalize(mean=SIXBAND_MEAN, std=SIXBAND_STD))
    
    return T.Compose(transforms)


class WildfireDataset(Dataset):
    """
    channels: "rgb" (3 bands) or "all" (6 bands)
    augment: apply augmentation (use True for training)
    cache: load all data into RAM (faster if you have enough memory)
    """
    RGB_IDX = [2, 1, 0]  # B04, B03, B02
    ALL_IDX = [0, 1, 2, 3, 4, 5]
    LABELS = {"fire": 0, "no_fire": 1, "burn_scar": 2}
    
    def __init__(self, split_dir, channels="rgb", augment=False, cache=False):
        self.idx = self.RGB_IDX if channels == "rgb" else self.ALL_IDX
        self.transform = get_transforms(train=augment, channels=channels)
        self.cache = cache
        self._cache = {}
        
        self.samples = []
        for label in ["fire", "no_fire", "burn_scar"]:
            d = Path(split_dir) / label
            if d.exists():
                for f in sorted(d.glob("*.npy")):  # sorted for reproducibility
                    if "_scl" not in f.name and "_mask" not in f.name:
                        self.samples.append((f, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, i):
        path, label = self.samples[i]
        
        # Check cache first
        if self.cache and i in self._cache:
            img = self._cache[i].clone()
        else:
            img = np.load(path).astype(np.float32)[:, :, self.idx]
            img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)  # handle bad values
            img = np.clip(img, 0, 1).transpose(2, 0, 1)
            img = torch.from_numpy(img)
            
            if self.cache:
                self._cache[i] = img.clone()
        
        if self.transform:
            img = self.transform(img)
        
        return img, self.LABELS[label]


def get_dataloaders(data_dir, channels="rgb", batch_size=32, num_workers=4, pin_memory=True, cache=False):
    """Create train/val/test dataloaders with proper augmentation."""
    data_dir = Path(data_dir)
    
    train = WildfireDataset(data_dir / "train", channels, augment=True, cache=cache)
    val = WildfireDataset(data_dir / "val", channels, augment=False, cache=cache)
    test = WildfireDataset(data_dir / "test", channels, augment=False, cache=cache)
    
    return (
        DataLoader(train, batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory),
        DataLoader(val, batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
        DataLoader(test, batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
    )
