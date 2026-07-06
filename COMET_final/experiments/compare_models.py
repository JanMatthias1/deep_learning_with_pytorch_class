import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

# --- DIRECTORIES ---
# 1. Baseline COMET Run
baseline_dir = "in_house_lnp_library_infer_results_withclsrep_OS/demo_in_house_lnp_data_overall_new_full_without_pbae_NPratios_updated_09222023_npratios_09252023gen_fig3dDeployTrain_allastest_infer_results/save_demo_in_house_ED09262023_fig3dDepolyTrain_fold_V0_lnp_np_finetune_contrastive-bs64-lr0.0001-lnpmodparams8-256-256-8-trainrat1-ep200-pat20-metricvalid_spearmanr_coeff-cagrad0.2-percentnoise0.1-labelmargin0.01-seed1_exp19"

# 2. Strategy A (Original GIN Fusion / Concatenation)
strat_a_dir = "./infer_results/infer_demo_in_house_ED09262023_fig3dii_fold_V0_lnp_np_finetune_contrastive-bs16-lr0.0001-lnpmodparams8-256-256-8-trainrat1-ep200-pat20-metricvalid_spearmanr_coeff-cagrad0.2-percentnoise0.1-labelmargin0.01-seed1_OS_GIN_fusion"

# 3. Strategy B (Gated Residual)
strat_b_dir = "./infer_results/infer_GIN_StrategyB_Run"

def get_data_from_dir(directory):
    if not os.path.exists(directory):
        return None
    pkl_file = next((f for f in os.listdir(directory) if f.endswith(".out.pkl")), None)
    if not pkl_file:
        return None
        
    with open(os.path.join(directory, pkl_file), "rb") as f:
        data = pickle.load(f)
        
    suffix = 'test' if any('test_target' in k for k in data.keys()) else 'infer'
    
    return {
        'b16_targ': np.array(data[f'in_house_lnp_B16F10_luc{suffix}_target']).flatten(),
        'b16_pred': np.array(data[f'in_house_lnp_B16F10_luc{suffix}_predict']).flatten(),
        'dc24_targ': np.array(data[f'in_house_lnp_DC24_luc{suffix}_target']).flatten(),
        'dc24_pred': np.array(data[f'in_house_lnp_DC24_luc{suffix}_predict']).flatten()
    }

baseline_data = get_data_from_dir(baseline_dir)
strat_a_data = get_data_from_dir(strat_a_dir)
strat_b_data = get_data_from_dir(strat_b_dir)

if baseline_data and strat_a_data and strat_b_data:
    print("\n" + "="*50)
    print("📊 3-WAY METRICS SUMMARY")
    print("="*50)
    
    datasets = [
        (baseline_data, "Baseline COMET"),
        (strat_a_data, "Strategy A (GIN Concatenation)"),
        (strat_b_data, "Strategy B (Gated Residual)")
    ]
    
    for data, model_name in datasets:
        print(f"\n--- {model_name} ---")
        for prefix, display_name in [('b16', 'B16F10'), ('dc24', 'DC2.4')]:
            targ = data[f'{prefix}_targ']
            pred = data[f'{prefix}_pred']
            spearman_corr, _ = spearmanr(pred, targ)
            pearson_corr, _ = pearsonr(pred, targ)
            print(f"{display_name} Cells: Spearman R: {spearman_corr:.4f} | Pearson R: {pearson_corr:.4f}")
            
    print("="*50 + "\n")

    # 3 Rows x 2 Columns Plot
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    fig.suptitle("Model Architecture Comparison", fontsize=16, fontweight='bold')
    
    for row_idx, (data, model_name) in enumerate(datasets):
        for col_idx, cell_line in enumerate([('b16', 'B16F10', 'blue'), ('dc24', 'DC2.4', 'green')]):
            prefix, display_name, color = cell_line
            targ = data[f'{prefix}_targ']
            pred = data[f'{prefix}_pred']
            
            corr, _ = spearmanr(pred, targ)
            
            ax = axes[row_idx][col_idx]
            ax.scatter(targ, pred, alpha=0.5, color=color, edgecolor='k')
            ax.set_title(f"[{model_name}]\n{display_name} Cells (Spearman R = {corr:.3f})", fontsize=12)
            ax.set_xlabel("True Target Value")
            ax.set_ylabel("Predicted Value")
            
            min_v, max_v = min(min(targ), min(pred)), max(max(targ), max(pred))
            ax.plot([min_v, max_v], [min_v, max_v], 'k--', lw=1.5, label='Ideal (y=x)')
            ax.legend()
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    save_path = "model_comparison_scatter_3way.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ Comparison plot saved successfully to: {os.path.abspath(save_path)}")
else:
    print("\n❌ Could not generate plot. One or more datasets failed to load.")