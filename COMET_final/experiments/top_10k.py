import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# The directory where the results were saved
results_dir = "in_house_lnp_library_infer_results_withclsrep_OS/demo_in_house_lnp_data_overall_new_full_without_pbae_NPratios_updated_09222023_npratios_09252023gen_fig3dDeployTrain_allastest_infer_results/save_demo_in_house_ED09262023_fig3dDepolyTrain_fold_V0_lnp_np_finetune_contrastive-bs64-lr0.0001-lnpmodparams8-256-256-8-trainrat1-ep200-pat20-metricvalid_spearmanr_coeff-cagrad0.2-percentnoise0.1-labelmargin0.01-seed1_exp19"

# Find the .out.pkl file
pkl_file_path = None
for file in os.listdir(results_dir):
    if file.endswith(".out.pkl"):
        pkl_file_path = os.path.join(results_dir, file)
        break

if pkl_file_path:
    with open(pkl_file_path, "rb") as f:
        data = pickle.load(f)
    
    # We will use the B16F10 cell line for this visualization
    targ = np.array(data['in_house_lnp_B16F10_lucinfer_target']).flatten()
    pred = np.array(data['in_house_lnp_B16F10_lucinfer_predict']).flatten()
    
    # --- Define "Hits" ---
    # Let's say a "True Hit" is any LNP in the Top 10% of actual lab results
    percentile_threshold = 90
    true_hit_threshold = np.percentile(targ, percentile_threshold)
    pred_hit_threshold = np.percentile(pred, percentile_threshold)
    
    # Create Binary Labels (1 = Hit, 0 = Dud)
    true_hits = (targ >= true_hit_threshold).astype(int)
    pred_hits = (pred >= pred_hit_threshold).astype(int)
    
    # --- PLOTTING ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Confusion Matrix (Top 10% vs Rest)
    cm = confusion_matrix(true_hits, pred_hits)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Predicted Failures', 'Predicted Hits'],
                yticklabels=['Actual Failures', 'Actual Hits'],
                annot_kws={"size": 14})
    axes[0].set_title("AI's Top 10% Predictions vs Reality", fontsize=14)
    
    # 2. Top-K Hit Rate (Enrichment)
    # If we take the top N predictions from the model, what % are actual hits?
    sort_indices = np.argsort(pred)[::-1] # Sort best to worst predictions
    sorted_true_hits = true_hits[sort_indices]
    
    # Evaluate at different screening budgets (Top 50, 100, 200, 500)
    budgets = [50, 100, 200, 500]
    hit_rates = []
    
    for k in budgets:
        hits_found = np.sum(sorted_true_hits[:k])
        hit_rates.append((hits_found / k) * 100)
        
    # Baseline random guessing rate (since True Hits are 10% of the dataset, random guess is 10%)
    baseline_rate = 10.0 
    
    axes[1].bar([str(b) for b in budgets], hit_rates, color='coral', edgecolor='black')
    axes[1].axhline(y=baseline_rate, color='red', linestyle='--', label='Random Guessing (10%)')
    axes[1].set_title("Hit Rate if Lab Synthesizes AI's Top Predictions", fontsize=14)
    axes[1].set_xlabel("Number of LNPs Tested (Lab Budget)", fontsize=12)
    axes[1].set_ylabel("Percentage that are Actual Hits (%)", fontsize=12)
    axes[1].set_ylim(0, 100)
    axes[1].legend()
    
    # Save the figure
    plt.tight_layout()
    save_path = "enrichment_metrics.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ Plot saved to: {os.path.abspath(save_path)}")

else:
    print("Could not find the .out.pkl file.")