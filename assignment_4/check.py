import os
import numpy as np

SPLITS_DIR = "/users/jmatthia/deep_learning/data/UCF50_splits"


def extract_group_id(file_path):
    clip_folder = os.path.basename(os.path.dirname(file_path))  # v_Basketball_g02_c01
    action_class = os.path.basename(os.path.dirname(os.path.dirname(file_path)))  # Basketball
    parts = clip_folder.split("_")
    if len(parts) >= 3 and parts[-2].startswith("g"):
        return (action_class, parts[-2])  # e.g. ('Basketball', 'g02')
    return ("unknown", "unknown_group")

def get_paths_from_dir(split_name):
    """Walks a split directory and returns all file paths."""
    split_path = os.path.join(SPLITS_DIR, split_name)
    paths = []
    for root, _, files in os.walk(split_path):
        for f in files:
            paths.append(os.path.join(root, f))
    return paths

def run_leakage_test():
    print("--- Running Data Leakage Test ---")

    train_paths = get_paths_from_dir("train")
    val_paths   = get_paths_from_dir("val")
    test_paths  = get_paths_from_dir("test")

    train_groups = set(extract_group_id(p) for p in train_paths)
    val_groups   = set(extract_group_id(p) for p in val_paths)
    test_groups  = set(extract_group_id(p) for p in test_paths)

    print(f"Total Unique Train Groups: {len(train_groups)}")
    print(f"Total Unique Val Groups:   {len(val_groups)}")
    print(f"Total Unique Test Groups:  {len(test_groups)}")

    train_val_overlap  = train_groups.intersection(val_groups)
    train_test_overlap = train_groups.intersection(test_groups)
    val_test_overlap   = val_groups.intersection(test_groups)

    leakage_found = False

    if train_val_overlap:
        print(f"\n[FAIL] Train/Val leakage — shared groups: {train_val_overlap}")
        leakage_found = True
    else:
        print("\n[PASS] No overlap between Train and Val.")

    if train_test_overlap:
        print(f"[FAIL] Train/Test leakage — shared groups: {train_test_overlap}")
        leakage_found = True
    else:
        print("[PASS] No overlap between Train and Test.")

    if val_test_overlap:
        print(f"[FAIL] Val/Test leakage — shared groups: {val_test_overlap}")
        leakage_found = True
    else:
        print("[PASS] No overlap between Val and Test.")

    if not leakage_found:
        print("\n[SUCCESS] All actors/environments cleanly isolated. Pipeline is leakage-free.")

if __name__ == "__main__":
    run_leakage_test()