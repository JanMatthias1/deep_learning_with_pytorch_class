import os
import pickle

# The directory where the results were saved
results_dir = "in_house_lnp_library_infer_results_withclsrep_OS/demo_in_house_lnp_data_overall_new_full_without_pbae_NPratios_updated_09222023_npratios_09252023gen_fig3dDeployTrain_allastest_infer_results/save_demo_in_house_ED09262023_fig3dDepolyTrain_fold_V0_lnp_np_finetune_contrastive-bs64-lr0.0001-lnpmodparams8-256-256-8-trainrat1-ep200-pat20-metricvalid_spearmanr_coeff-cagrad0.2-percentnoise0.1-labelmargin0.01-seed1_exp19"

# Find the .out.pkl file in the directory
pkl_file_path = None
for file in os.listdir(results_dir):
    if file.endswith(".out.pkl"):
        pkl_file_path = os.path.join(results_dir, file)
        break

if pkl_file_path:
    print(f"Loading results from: {pkl_file_path}\n")
    
    # Load the pickle file
    with open(pkl_file_path, "rb") as f:
        data = pickle.load(f)
    
    # Display what type of data it is and its length
    print(f"Data type: {type(data)}")
    if isinstance(data, list) or isinstance(data, dict):
        print(f"Number of entries: {len(data)}")
    
    print("-" * 50)
    
    # Print a preview of the data (first 3 items if it's a list, or the keys if it's a dict)
    if isinstance(data, list):
        print("Preview of the first 3 entries:")
        for i, item in enumerate(data[:3]):
            print(f"\nEntry {i+1}:")
            print(item)
    elif isinstance(data, dict):
        print("Keys in the dictionary:")
        print(list(data.keys())[:10]) # preview first 10 keys
        
        # Optionally print the first key-value pair
        first_key = list(data.keys())[0]
        print(f"\nPreview of first entry (Key: {first_key}):")
        print(data[first_key])
else:
    print("Could not find a .out.pkl file in the specified directory.")
