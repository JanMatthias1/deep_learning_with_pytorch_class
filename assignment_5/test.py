"""Evaluate trained Swin-B on CIFAR-100 test set."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from torchvision.models import swin_b

from data import get_dataloaders

def build_model(num_classes=100):
    """Build Swin-B with a fresh classification head."""
    model = swin_b(weights=None)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    return model


def get_predictions(model, loader, device):
    """Return ground-truth labels, predicted labels, and class probabilities."""
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            all_labels.append(labels.numpy())
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    return all_labels, all_preds, all_probs


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path="confusion_matrix.png"):
    """Plot and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(20, 20))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix (100 classes)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if class_names is not None:
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=90, fontsize=5)
        ax.set_yticklabels(class_names, fontsize=5)
    fig.colorbar(im, ax=ax, shrink=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def main(cli_args):
    """Load checkpoint and run full evaluation on the test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    _, _, test_loader = get_dataloaders(
        data_root=cli_args.data_root,
        batch_size=cli_args.batch_size,
        num_workers=cli_args.num_workers,
    )

    class_names = test_loader.dataset.classes

    model = build_model(num_classes=100).to(device)
    model.load_state_dict(torch.load(cli_args.checkpoint, map_location=device))
    print(f"Loaded checkpoint: {cli_args.checkpoint}")

    y_true, y_pred, y_probs = get_predictions(model, test_loader, device)

    acc = accuracy_score(y_true, y_pred)
    print(f"\n{'='*60}")
    print(f"Overall Test Accuracy: {acc:.4f}")
    print(f"{'='*60}")

    print("\nClassification Report (per-class + macro avg):")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    os.makedirs(cli_args.output_dir, exist_ok=True)
    cm_path = os.path.join(cli_args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, class_names=class_names, save_path=cm_path)

    roc_auc = roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
    print(f"ROC-AUC (one-vs-rest, macro): {roc_auc:.4f}")

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  Accuracy:           {acc:.4f}")
    print(f"  ROC-AUC (ovr):      {roc_auc:.4f}")
    print(f"  Confusion matrix:   {cm_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Swin-B on CIFAR-100 test set")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    main(args)
