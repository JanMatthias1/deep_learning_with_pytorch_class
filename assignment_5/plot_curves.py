import re
import argparse
import matplotlib.pyplot as plt

def parse_log(log_path):
    epochs, train_loss, train_acc, val_loss, val_acc = [], [], [], [], []
    pattern = re.compile(
        r"Epoch\s+(\d+)/\d+.*?train loss ([\d.]+) acc ([\d.]+).*?val loss ([\d.]+) acc ([\d.]+)"
    )
    with open(log_path, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epochs.append(int(m.group(1)))
                train_loss.append(float(m.group(2)))
                train_acc.append(float(m.group(3)))
                val_loss.append(float(m.group(4)))
                val_acc.append(float(m.group(5)))
    return epochs, train_loss, train_acc, val_loss, val_acc


def plot(epochs, train_loss, train_acc, val_loss, val_acc, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    axes[0].plot(epochs, train_loss, label="Train loss")
    axes[0].plot(epochs, val_loss, label="Val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss curves")
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy
    axes[1].plot(epochs, train_acc, label="Train acc")
    axes[1].plot(epochs, val_acc, label="Val acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy curves")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="Path to the .out log file")
    parser.add_argument("--out", default="curves.png", help="Output image path")
    args = parser.parse_args()

    epochs, train_loss, train_acc, val_loss, val_acc = parse_log(args.log)
    if not epochs:
        print("No epoch lines found. Check the log format.")
    else:
        print(f"Parsed {len(epochs)} epochs (1-{epochs[-1]})")
        plot(epochs, train_loss, train_acc, val_loss, val_acc, args.out)
