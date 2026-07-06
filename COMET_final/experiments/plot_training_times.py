import matplotlib.pyplot as plt
import numpy as np

# --- 1. The Data ---
models = ['Baseline COMET', 'GIN\n(2D)', 'SchNet\n(3D)', '3D Infomax\n(3D)']
epochs = [135, 85, 140, 40]
total_time = [32.77, 20.32, 33.38, 6.35]
time_per_epoch = [9.51, 9.52, 9.52, 9.52]

# --- 2. Styling (Publication Ready) ---
# Soft, professional color palette (Grey for baseline, Blue/Green/Red for experiments)
colors = ['#B0B0B0', '#4C72B0', '#55A868', '#C44E52']

fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
fig.suptitle('Computational Efficiency & Convergence of Auxiliary Embeddings', 
             fontsize=16, fontweight='bold', y=1.02)

# Helper function to auto-label the top of the bars
def add_value_labels(ax, values, format_str="%.2f"):
    for i, v in enumerate(values):
        label = format_str % v if isinstance(v, float) else str(v)
        ax.text(i, v + (max(values) * 0.03), label, ha='center', va='bottom', fontsize=11, fontweight='bold')

# --- Panel A: Epochs Run ---
axes[0].bar(models, epochs, color=colors, edgecolor='black', linewidth=1.2)
axes[0].set_title('Time to Convergence\n(Total Epochs Run)', fontsize=14)
axes[0].set_ylabel('Number of Epochs', fontsize=12)
axes[0].set_ylim(0, max(epochs) * 1.15)
add_value_labels(axes[0], epochs, "%d")

# --- Panel B: Total Time ---
axes[1].bar(models, total_time, color=colors, edgecolor='black', linewidth=1.2)
axes[1].set_title('Total Training Time\n(Minutes)', fontsize=14)
axes[1].set_ylabel('Minutes', fontsize=12)
axes[1].set_ylim(0, max(total_time) * 1.15)
add_value_labels(axes[1], total_time, "%.2f")

# --- Panel C: Time per Epoch ---
axes[2].bar(models, time_per_epoch, color=colors, edgecolor='black', linewidth=1.2)
axes[2].set_title('Hardware Overhead\n(Time per Epoch)', fontsize=14)
axes[2].set_ylabel('Seconds per Epoch', fontsize=12)
axes[2].set_ylim(0, max(time_per_epoch) * 1.2)
add_value_labels(axes[2], time_per_epoch, "%.2f")

# --- 3. Clean up the aesthetics ---
for ax in axes:
    # Rotate x-axis labels slightly for readability
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=25, ha='right', fontsize=11)
    
    # Add a subtle grid behind the bars
    ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
    for bar in ax.patches:
        bar.set_zorder(3) # Ensure bars render in front of the grid
        
    # Remove top and right borders (standard for scientific papers)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()

# Save at 300 DPI (High Resolution for Papers/Presentations)
save_path = 'training_efficiency_3panel.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✅ Publication-ready figure saved successfully to: {save_path}")
