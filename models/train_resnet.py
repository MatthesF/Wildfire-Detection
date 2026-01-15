
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models

from dataset import get_dataloaders

# ============================================
# CONFIG
# ============================================
EXPERIMENT = "rgb"  # "rgb" (3 channels) or "all" (6 channels)

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "splits"
OUTPUT_DIR = ROOT / "models" / "checkpoints"

NUM_CLASSES = 3
BATCH_SIZE = 32
NUM_EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
PATIENCE = 5

SAVE_EVERY = 5  # save checkpoint every N epochs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_resnet(num_channels, num_classes):
    """Create ResNet-50, adapting first conv for non-RGB input if needed."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    
    if num_channels != 3:
        old = model.conv1
        new = nn.Conv2d(num_channels, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
        with torch.no_grad():
            new.weight[:, :3] = old.weight
            if num_channels > 3:
                mean_w = old.weight.mean(dim=1, keepdim=True)
                for i in range(3, num_channels):
                    new.weight[:, i:i+1] = mean_w
        model.conv1 = new
    
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def run_epoch(model, loader, criterion, optimizer=None):
    """Train or eval one epoch. Pass optimizer=None for eval mode."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    
    total_loss, correct, total = 0.0, 0, 0
    preds_all, labels_all = [], []
    
    ctx = torch.no_grad() if not is_train else torch.enable_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            out = model(imgs)
            loss = criterion(out, labels)
            
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item() * imgs.size(0)
            pred = out.argmax(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            preds_all.extend(pred.cpu().tolist())
            labels_all.extend(labels.cpu().tolist())
    
    return total_loss / total, correct / total, preds_all, labels_all


def train():
    """Main training loop."""
    exp_dir = OUTPUT_DIR / EXPERIMENT
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    channels = "rgb" if EXPERIMENT == "rgb" else "all"
    num_ch = 3 if EXPERIMENT == "rgb" else 6
    
    train_loader, val_loader, test_loader = get_dataloaders(DATA_DIR, channels, BATCH_SIZE, NUM_WORKERS)
    
    model = create_resnet(num_ch, NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    
    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    print(f"Training {EXPERIMENT} | {num_ch} channels | {DEVICE}")
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")
    
    no_improve = 0
    for epoch in range(NUM_EPOCHS):
        t0 = time.time()
        
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = run_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        improved = val_acc > best_acc
        if improved:
            best_acc = val_acc
            no_improve = 0
            torch.save({"epoch": epoch, "model": model.state_dict(), "acc": best_acc}, exp_dir / "best.pth")
        else:
            no_improve += 1
        
        # Save checkpoint every N epochs
        if (epoch + 1) % SAVE_EVERY == 0:
            torch.save({"epoch": epoch, "model": model.state_dict(), "acc": val_acc}, 
                       exp_dir / f"epoch_{epoch+1}.pth")
        
        print(f"Epoch {epoch+1:2d} | train {train_acc:.3f} | val {val_acc:.3f} | {time.time()-t0:.0f}s" + (" *" if improved else ""))
        
        if no_improve >= PATIENCE:
            print("Early stop")
            break
    
    # Test
    ckpt = torch.load(exp_dir / "best.pth", map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    test_loss, test_acc, preds, labels = run_epoch(model, test_loader, criterion)
    
    class_names = ["fire", "no_fire", "burn_scar"]
    per_class = {}
    for i, name in enumerate(class_names):
        mask = [l == i for l in labels]
        if sum(mask) > 0:
            correct = sum(p == l for p, l, m in zip(preds, labels, mask) if m)
            per_class[name] = correct / sum(mask)
    
    print(f"\nTest accuracy: {test_acc:.3f}")
    for name, acc in per_class.items():
        print(f"  {name}: {acc:.3f}")
    
    results = {
        "experiment": EXPERIMENT, "test_acc": test_acc, "best_val_acc": best_acc,
        "per_class": per_class, "history": history, "timestamp": datetime.now().isoformat()
    }
    with open(exp_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    train()
