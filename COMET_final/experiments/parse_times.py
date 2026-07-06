import os
import re

# The log files we want to compare
log_files = {
    "Baseline COMET": "baseline_output.log", 
    "GIN (Baseline)": "gin_output.log",
    "SchNet 3D": "schnet_output.log",
    "3D Infomax": "infomax_output.log"
}

def parse_log(log_path):
    stats = {
        "Epochs Run": 0,
        "Total Time (min)": "N/A",
        "Time/Epoch (sec)": "N/A"
    }
    
    if not os.path.exists(log_path):
        return stats

    epochs = []
    epoch_times = []
    total_time_sec = 0.0

    with open(log_path, 'r') as f:
        for line in f:
            # 1. Extract Epochs
            epoch_match = re.search(r'end of epoch (\d+)', line)
            if epoch_match:
                epochs.append(int(epoch_match.group(1)))
            
            # 2. Extract Wall Time per Epoch (Fairseq/UniMol logs this as wall=...)
            wall_match = re.search(r'wall=([\d.]+)', line)
            if wall_match:
                epoch_times.append(float(wall_match.group(1)))
                
            # 3. Extract Explicit Total Training Time (if available)
            time_match = re.search(r'done training in ([\d.]+) seconds', line)
            if time_match:
                total_time_sec = float(time_match.group(1))

    # Calculate final stats
    if epochs:
        stats["Epochs Run"] = max(epochs)
        
    # Robust Fallback Calculation
    if epoch_times:
        # Average the wall time per epoch
        avg_time_per_epoch = sum(epoch_times) / len(epoch_times)
        stats["Time/Epoch (sec)"] = round(avg_time_per_epoch, 2)
        
        # If the "done training" line was missing (like in Infomax), calculate it manually!
        if total_time_sec == 0:
            total_time_sec = avg_time_per_epoch * stats["Epochs Run"]
            
    if total_time_sec > 0:
        stats["Total Time (min)"] = round(total_time_sec / 60, 2)
        
    return stats

# --- Generate the Table ---
print("\n" + "="*80)
print(f"{'Model Architecture':<25} | {'Epochs Run':<12} | {'Total Time (min)':<18} | {'Time/Epoch (sec)':<15}")
print("-" * 80)

for model_name, file_name in log_files.items():
    if os.path.exists(file_name):
        s = parse_log(file_name)
        print(f"{model_name:<25} | {str(s['Epochs Run']):<12} | {str(s['Total Time (min)']):<18} | {str(s['Time/Epoch (sec)']):<15}")
    else:
        print(f"{model_name:<25} | {'[Log file not found]':<52}")

print("="*80 + "\n")