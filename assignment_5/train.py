"""Train Swin-B from scratch on CIFAR-100."""

import argparse
import math
import os
import time

import torch
from torch import nn
from torch.optim import AdamW
from torchvision.models import swin_b
import wandb

from data import get_dataloaders

def build_model(num_classes=100):
    """
    Swin-Base initialized from scratch (weights=None).

    Swin-base architecture from: https://arxiv.org/abs/2103.14030, which is implemented in Pytorch
    """
    model = swin_b(weights=None)
    in_features = model.head.in_features  # 768
    model.head = nn.Linear(in_features, num_classes)
    # Reinit the new head
    nn.init.trunc_normal_(model.head.weight, std=0.02)
    nn.init.zeros_(model.head.bias)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Train the model for one epoch and return the average loss and accuracy.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    """Evaluate the model on the given data loader and return the average loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total

def main(cli_args):  # pylint: disable=too-many-locals
    """
    Main training loop for Swin-B on CIFAR-100.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Data ----
    train_loader, val_loader, _ = get_dataloaders(
        data_root=cli_args.data_root,
        batch_size=cli_args.batch_size,
        num_workers=cli_args.num_workers,
    )

    # ---- Model ---- --> swin tranaformer here
    model = build_model(num_classes=100).to(device)
    print(f"Swin-B parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Loss, optimizer, scheduler ----
    criterion = nn.CrossEntropyLoss()

    def lr_fn(epoch):
        """Warmup + cosine annealing learning rate schedule."""
        if epoch < cli_args.warmup_epochs:
            return (epoch + 1) / cli_args.warmup_epochs
        progress = (epoch - cli_args.warmup_epochs) / (cli_args.epochs - cli_args.warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    optimizer = AdamW(model.parameters(), lr=cli_args.lr, weight_decay=cli_args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_fn)

    # ---- Training loop ----
    best_val_acc = 0.0
    patience_counter = 0
    os.makedirs(cli_args.save_dir, exist_ok=True)

    wandb.init(
        project="CIFAR100-SwinB-Training",
        config={
            "model_type": "swin_b",
            "n_classes": 100,
            "batch_size": cli_args.batch_size,
            "learning_rate": cli_args.lr,
            "weight_decay": cli_args.weight_decay,
            "n_epochs": cli_args.epochs,
            "patience": cli_args.patience,
        }
    )

    for epoch in range(1, cli_args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        wandb.log({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch": epoch,
        })

        print(
            f"Epoch {epoch:3d}/{cli_args.epochs} | "
            f"lr {optimizer.param_groups[0]['lr']:.2e} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"{time.time() - t0:.1f}s"
        )

        # Checkpoint best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(cli_args.save_dir, "best_model.pt"))
            print(f"  -> New best val acc: {val_acc:.4f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= cli_args.patience:
            print(f"Early stopping at epoch {epoch} no improvement for {cli_args.patience} epochs")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Swin-B from scratch on CIFAR-100")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    main(args)
