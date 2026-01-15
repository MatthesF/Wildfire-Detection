import json
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader


class WildfireDataset(Dataset):
    """
    Load wildfire patches from split directory.
    
    channels: "rgb" (3 bands) or "all" (6 bands)
    """
    RGB_IDX = [2, 1, 0]  # B04, B03, B02
    ALL_IDX = [0, 1, 2, 3, 4, 5]
    LABELS = {"fire": 0, "no_fire": 1, "burn_scar": 2}
    
    def __init__(self, split_dir, channels="rgb", transform=None):
        self.transform = transform
        self.idx = self.RGB_IDX if channels == "rgb" else self.ALL_IDX
        
        self.samples = []
        for label in ["fire", "no_fire", "burn_scar"]:
            d = Path(split_dir) / label
            if d.exists():
                for f in d.glob("*.npy"):
                    if "_scl" not in f.name and "_mask" not in f.name:
                        self.samples.append((f, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, i):
        path, label = self.samples[i]
        img = np.load(path).astype(np.float32)[:, :, self.idx]
        img = np.clip(img, 0, 1).transpose(2, 0, 1)
        img = torch.from_numpy(img)
        if self.transform:
            img = self.transform(img)
        return img, self.LABELS[label]


def get_dataloaders(data_dir, channels="rgb", batch_size=32, num_workers=4):
    """Create train/val/test dataloaders."""
    data_dir = Path(data_dir)
    
    train = WildfireDataset(data_dir / "train", channels)
    val = WildfireDataset(data_dir / "val", channels)
    test = WildfireDataset(data_dir / "test", channels)
    
    return (
        DataLoader(train, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(val, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        DataLoader(test, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    )
