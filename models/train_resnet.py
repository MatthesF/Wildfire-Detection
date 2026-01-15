import json
import random
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score

from dataset import get_dataloaders

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# ============================================
# CONFIG
# ============================================
EXPERIMENT = "all"  # "rgb" (3 channels) or "all" (6 channels)
USE_WANDB = True
MODEL = "resnet18"  # "resnet18" or "resnet50"
# Note: Using ReduceLROnPlateau for adaptive LR decay based on val_acc

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "splits"
OUTPUT_DIR = ROOT / "models" / "checkpoints"

NUM_CLASSES = 3
CLASS_NAMES = ["fire", "no_fire", "burn_scar"]
BATCH_SIZE = 16
NUM_EPOCHS = 15
LR = 2.6e-4
WEIGHT_DECAY = 1.0e-5
NUM_WORKERS = 0  # 0 is often better for MPS
PATIENCE = 3
SAVE_EVERY = 5
DROPOUT = 0.15
LABEL_SMOOTHING = 0.12
CACHE_DATA = True  # load all data into RAM (faster if you have enough memory)
SEED = 42

# Device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    PIN_MEMORY = False
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    PIN_MEMORY = True
else:
    DEVICE = torch.device("cpu")
    PIN_MEMORY = False


def seed_everything(seed):
    """Seed all random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_resnet(num_channels, num_classes, model_name="resnet18", dropout=0.0):
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
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
    
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(num_features, num_classes)
    )
    return model


def run_epoch(model, loader, criterion, optimizer=None, desc=""):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    
    total_loss, correct, total = 0.0, 0, 0
    preds_all, labels_all = [], []
    
    pbar = tqdm(loader, desc=desc, leave=False)
    ctx = torch.no_grad() if not is_train else torch.enable_grad()
    
    with ctx:
        for imgs, labels in pbar:
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
            
            pbar.set_postfix(loss=f"{total_loss/total:.3f}", acc=f"{correct/total:.3f}")
            
            preds_all.extend(pred.cpu().tolist())
            labels_all.extend(labels.cpu().tolist())
    
    return total_loss / total, correct / total, preds_all, labels_all


def per_class_accuracy(preds, labels, num_classes):
    """Compute per-class accuracy."""
    total = [0] * num_classes
    correct = [0] * num_classes
    for p, l in zip(preds, labels):
        total[l] += 1
        if p == l:
            correct[l] += 1
    return [correct[i] / total[i] if total[i] else 0.0 for i in range(num_classes)]


def print_metrics(preds, labels, class_names):
    """Print detailed classification metrics."""
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=class_names, digits=3))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(labels, preds)
    # Pretty print
    header = "           " + "  ".join(f"{n:>10}" for n in class_names)
    print(header)
    for i, row in enumerate(cm):
        print(f"{class_names[i]:>10} " + "  ".join(f"{v:>10}" for v in row))


def train():
    seed_everything(SEED)
    
    exp_dir = OUTPUT_DIR / EXPERIMENT
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    channels = "rgb" if EXPERIMENT == "rgb" else "all"
    num_ch = 3 if EXPERIMENT == "rgb" else 6
    
    print(f"\n{'='*50}")
    print(f"Wildfire Detection - {EXPERIMENT.upper()}")
    print(f"{'='*50}")
    print(f"Model: {MODEL}")
    print(f"Device: {DEVICE}")
    print(f"Channels: {num_ch}")
    print(f"Dropout: {DROPOUT} | Label smoothing: {LABEL_SMOOTHING}")
    print(f"Cache data: {CACHE_DATA} | Seed: {SEED}")
    print(f"{'='*50}\n")
    
    # Initialize wandb
    if USE_WANDB and HAS_WANDB:
        wandb.init(
            project="wildfire-detection",
            name=f"{EXPERIMENT}_{MODEL}_{datetime.now().strftime('%H%M%S')}",
            config={
                "experiment": EXPERIMENT, "model": MODEL, "channels": num_ch,
                "batch_size": BATCH_SIZE, "lr": LR, "dropout": DROPOUT,
                "label_smoothing": LABEL_SMOOTHING, "seed": SEED
            }
        )
    
    # Data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        DATA_DIR, channels, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, cache=CACHE_DATA
    )
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}\n")
    
    # Model
    print(f"Loading {MODEL}...")
    model = create_resnet(num_ch, NUM_CLASSES, model_name=MODEL, dropout=DROPOUT).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6)
    
    print("Starting training...\n")
    
    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    no_improve = 0
    
    for epoch in range(NUM_EPOCHS):
        t0 = time.time()
        
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, criterion, optimizer, f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        val_loss, val_acc, val_preds, val_labels = run_epoch(model, val_loader, criterion, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]")
        val_per_class = per_class_accuracy(val_preds, val_labels, NUM_CLASSES)
        val_pc_dict = {CLASS_NAMES[i]: val_per_class[i] for i in range(NUM_CLASSES)}
        
        # Step scheduler based on val_acc
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        improved = val_acc > best_acc
        if improved:
            best_acc = val_acc
            no_improve = 0
            # Save full checkpoint
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
                "config": {"experiment": EXPERIMENT, "model": MODEL, "channels": num_ch}
            }, exp_dir / "best.pth")
        else:
            no_improve += 1
        
        if (epoch + 1) % SAVE_EVERY == 0:
            torch.save({"epoch": epoch, "model": model.state_dict(), "acc": val_acc}, exp_dir / f"epoch_{epoch+1}.pth")
        
        if USE_WANDB and HAS_WANDB:
            wandb.log({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_fire": val_pc_dict["fire"],
                "val_no_fire": val_pc_dict["no_fire"],
                "val_burn_scar": val_pc_dict["burn_scar"],
                "lr": current_lr,
            })
        
        status = " [NEW BEST]" if improved else ""
        val_pc_text = " | ".join(f"{k}:{v:.3f}" for k, v in val_pc_dict.items())
        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | Train: {train_acc:.3f} | Val: {val_acc:.3f} | Best: {best_acc:.3f} | LR: {current_lr:.2e} | {time.time()-t0:.0f}s{status}")
        print(f"  Val per-class -> {val_pc_text}")
        
        if no_improve >= PATIENCE:
            print(f"\nEarly stopping (no improvement for {PATIENCE} epochs)")
            break
    
    # Test
    print(f"\n{'='*50}")
    print("Testing best model...")
    print(f"{'='*50}")
    
    ckpt = torch.load(exp_dir / "best.pth", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_loss, test_acc, preds, labels = run_epoch(model, test_loader, criterion, desc="Testing")
    
    class_names = ["fire", "no_fire", "burn_scar"]
    
    print(f"\nTest Accuracy: {test_acc:.3f}")
    print_metrics(preds, labels, class_names)
    
    # Per-class accuracy for wandb
    per_class = {}
    for i, name in enumerate(class_names):
        mask = [l == i for l in labels]
        if sum(mask) > 0:
            correct = sum(p == l for p, l, m in zip(preds, labels, mask) if m)
            per_class[name] = correct / sum(mask)
    
    if USE_WANDB and HAS_WANDB:
        wandb.log({"test_acc": test_acc, **{f"test_{k}": v for k, v in per_class.items()}})
        wandb.finish()
    
    results = {
        "experiment": EXPERIMENT, "model": MODEL, "test_acc": test_acc, "best_val_acc": best_acc,
        "per_class": per_class, "history": history, "seed": SEED
    }
    with open(exp_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {exp_dir}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    train()
