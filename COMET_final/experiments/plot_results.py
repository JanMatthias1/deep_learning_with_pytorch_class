import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Updated directory pointing to your new GIN fusion run
results_dir = "infer_results/infer_demo_in_house_ED09262023_fig3dii_fold_V0_lnp_np_finetune_contrastive-bs16-lr0.0001-lnpmodparams8-256-256-8-trainrat1-ep200-pat20-metricvalid_spearmanr_coeff-cagrad0.2-percentnoise0.1-labelmargin0.01-seed1_OS_GIN_fusion"

# Find the .out.pkl file
pkl_file_path = None
for file in os.listdir(results_dir):
    if file.endswith(".out.pkl"):
        pkl_file_path = os.path.join(results_dir, file)
        break

if pkl_file_path:
    print(f"Loading results from: {pkl_file_path}")
    with open(pkl_file_path, "rb") as f:
        data = pickle.load(f)
    
    # Extract True Targets and Model Predictions using the 'test' subset keys
    b16_targ = np.array(data['in_house_lnp_B16F10_luctest_target']).flatten()
    b16_pred = np.array(data['in_house_lnp_B16F10_luctest_predict']).flatten()
    
    dc24_targ = np.array(data['in_house_lnp_DC24_luctest_target']).flatten()
    dc24_pred = np.array(data['in_house_lnp_DC24_luctest_predict']).flatten()
    
    # Calculate Spearman Correlation
    b16_corr, _ = spearmanr(b16_pred, b16_targ)
    dc24_corr, _ = spearmanr(dc24_pred, dc24_targ)
    
    # Create the Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # B16F10 Cell Line Plot
    axes[0].scatter(b16_targ, b16_pred, alpha=0.5, color='blue', edgecolor='k')
    axes[0].set_title(f"B16F10 Cells (Spearman R = {b16_corr:.3f})", fontsize=14)
    axes[0].set_xlabel("True Target Value", fontsize=12)
    axes[0].set_ylabel("Predicted Value", fontsize=12)
    
    # Perfect prediction y=x line
    min_val = min(min(b16_targ), min(b16_pred))
    max_val = max(max(b16_targ), max(b16_pred))
    axes[0].plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5, label='Ideal (y=x)')
    axes[0].legend()

    # DC2.4 Cell Line Plot
    axes[1].scatter(dc24_targ, dc24_pred, alpha=0.5, color='green', edgecolor='k')
    axes[1].set_title(f"DC2.4 Cells (Spearman R = {dc24_corr:.3f})", fontsize=14)
    axes[1].set_xlabel("True Target Value", fontsize=12)
    axes[1].set_ylabel("Predicted Value", fontsize=12)
    
    min_val2 = min(min(dc24_targ), min(dc24_pred))
    max_val2 = max(max(dc24_targ), max(dc24_pred))
    axes[1].plot([min_val2, max_val2], [min_val2, max_val2], 'k--', lw=1.5, label='Ideal (y=x)')
    axes[1].legend()

    # Save the figure with a distinct name
    plt.tight_layout()
    save_path = "accuracy_scatter_plot_GIN_fusion.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ Plot saved successfully to: {os.path.abspath(save_path)}")

else:
    print(f"Could not find the .out.pkl file in {results_dir}")