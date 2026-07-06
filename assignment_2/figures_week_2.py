import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import ConfusionMatrixDisplay


def plot_cm_only(cm, title="Confusion Matrix", normalize=False):
    cm = np.asarray(cm)

    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    # show tick labels as class indices
    n = cm.shape[0]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(range(n))
    ax.set_yticklabels(range(n))

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

def last_seed_for_model(model_type, split="VAL", confusion_matrix_dictionary=None):
    seeds = []
    for k in confusion_matrix_dictionary.keys():
        if k.startswith(model_type) and k.endswith(f"_{split}"):
            seeds.append(int(k.split("_seed_")[1].split("_")[0]))
    return max(seeds)

def plot_cm_grid(split, normalize, model_types, confusion_matrix_dictionary, name_map, class_names=None):

    fig, axes = plt.subplots(1, len(model_types), figsize=(26, 5), constrained_layout=True)
    
    ims = []

    for i, mt in enumerate(model_types):
        s = last_seed_for_model(mt, split, confusion_matrix_dictionary)
        key = f"{mt}_seed_{s}_{split}"

        cm = np.asarray(confusion_matrix_dictionary[key])
        if normalize:
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        ax = axes[i]
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
        ims.append(im)

        ax.set_title(f"{name_map.get(mt, mt)} (seed {s})", fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        n = cm.shape[0]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))

        if class_names is not None:
            ax.set_xticklabels(class_names, rotation=60, ha="right", fontsize=8)
            ax.set_yticklabels(class_names, fontsize=8)
        else:
            ax.set_xticklabels(range(n), fontsize=8)
            ax.set_yticklabels(range(n), fontsize=8)

    # Proper single colorbar on the right
    cbar = fig.colorbar(ims[0], ax=axes, shrink=0.8, location="right")
    cbar.set_label("Proportion" if normalize else "Count")

    if split == "VAL":
        fig.suptitle(
            f"Best {split} Confusion Matrices" + (" (row-normalized)" if normalize else ""),
            fontsize=14
        )
    else:
        fig.suptitle(
            f"{split} Confusion Matrices" + (" (row-normalized)" if normalize else ""),
            fontsize=14
        )

    return fig

def val_test_top_1_accuracy(results, results_test):
    
    name_map = {
        "model_scratch": "S0",
        "imagenet_pretrained_frozen": "I-Frozen",
        "imagenet_pretrained_fine_tune": "I-FT",
        "ssl_pretrained_fine_tune_frozen": "SSL-Frozen",
        "ssl_pretrained_fine_tune_full": "SSL-FT"}

    # ---- Best validation epoch per seed ----
    best_val = results.loc[
        results.groupby(["Model_Type","Seed"])["Val_top1"].idxmax()
    ]

    # ---- Aggregate validation (mean + sd) ----
    val_summary = (
        best_val
        .groupby("Model_Type")[["Val_top1","Val_Loss"]]
        .agg(["mean","std"])
    )

    # flatten multi-index columns
    val_summary.columns = ["_".join(col) for col in val_summary.columns]
    val_summary = val_summary.reset_index()

    # ---- Aggregate test (mean + sd) ----
    test_summary = (
        results_test
        .groupby("Model_Type")[["Test_Acc","Test_Loss"]]
        .agg(["mean","std"])
    )

    test_summary.columns = ["_".join(col) for col in test_summary.columns]
    test_summary = test_summary.reset_index()

    # ---- Merge ----
    table = val_summary.merge(test_summary, on="Model_Type")

    # map names
    table["Condition"] = table["Model_Type"].map(name_map)

    # ---- Format mean ± sd strings ----
    table["Val Top-1"] = table.apply(
        lambda x: f"{x.Val_top1_mean:.4f} ± {x.Val_top1_std:.4f}", axis=1
    )

    table["Val Loss"] = table.apply(
        lambda x: f"{x.Val_Loss_mean:.4f} ± {x.Val_Loss_std:.4f}", axis=1
    )

    table["Test Top-1"] = table.apply(
        lambda x: f"{x.Test_Acc_mean:.4f} ± {x.Test_Acc_std:.4f}", axis=1
    )

    table["Test Loss"] = table.apply(
        lambda x: f"{x.Test_Loss_mean:.4f} ± {x.Test_Loss_std:.4f}", axis=1
    )

    # final clean table
    final_table = table[[
        "Condition",
        "Val Top-1",
        "Val Loss",
        "Test Top-1",
        "Test Loss"
    ]].sort_values("Condition")

    print(final_table)

def figure_train_validation_curves(results, metric="accuracy"):

    # ---- Select metric columns ----
    if metric == "accuracy":
        train_col = "Train_top1"
        val_col   = "Val_top1"
        y_label   = "Top-1 accuracy"
        title_tag = "Accuracy"
    elif metric == "loss":
        train_col = "Train_Loss"
        val_col   = "Val_Loss"
        y_label   = "Loss"
        title_tag = "Loss"
    else:
        raise ValueError("metric must be 'accuracy' or 'loss'")

    # ---- Aggregate mean and std across seeds ----
    def agg_mean_std(df, metric_col):
        return (
            df.groupby(["Model_Type", "Epoch"])[metric_col]
            .agg(["mean", "std"])
            .reset_index()
        )

    train = agg_mean_std(results, train_col)
    val   = agg_mean_std(results, val_col)

    models = sorted(results["Model_Type"].unique())

    # ---- Name mapping ----
    name_map = {
        "model_scratch": "ResNet18 Scratch",
        "imagenet_pretrained_frozen": "I-Frozen",
        "imagenet_pretrained_fine_tune": "I-Full",
        "ssl_pretrained_fine_tune_frozen": "SSL-Frozen",
        "ssl_pretrained_fine_tune_full": "SSL-Full",
    }

    # ---- Color palette ----
    cmap = plt.get_cmap("Dark2")
    colors = {m: cmap(i) for i, m in enumerate(models)}

    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True)

    # ---- Train ----
    ax = axes[0]
    for m in models:
        d = train[train["Model_Type"] == m].sort_values("Epoch")

        ax.plot(d["Epoch"], d["mean"],
                color=colors[m],
                linewidth=2.5)

        ax.fill_between(d["Epoch"],
                        d["mean"] - d["std"],
                        d["mean"] + d["std"],
                        color=colors[m],
                        alpha=0.20,
                        linewidth=0)

    ax.set_title(f"Train {title_tag} (mean ± SD)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)

    # ---- Validation ----
    ax = axes[1]
    for m in models:
        d = val[val["Model_Type"] == m].sort_values("Epoch")

        ax.plot(d["Epoch"], d["mean"],
                color=colors[m],
                linewidth=2.5)

        ax.fill_between(d["Epoch"],
                        d["mean"] - d["std"],
                        d["mean"] + d["std"],
                        color=colors[m],
                        alpha=0.20,
                        linewidth=0)

    ax.set_title(f"Validation {title_tag} (mean ± SD)")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.25)

    # ---- Bottom legend ----
    legend_handles = [
        Line2D([0], [0],
               marker='o',
               color='w',
               label=name_map[m],
               markerfacecolor=colors[m],
               markersize=8)
        for m in models
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(models),
        frameon=False
    )

    fig.tight_layout(rect=[0, 0.10, 1, 1])
    return fig
