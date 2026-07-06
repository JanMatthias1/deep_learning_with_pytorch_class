"""Data extraction and train/val/test splitting for the UCF50 dataset."""
import os
import re
import sys
import shutil
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit

sys.path.append("/users/jmatthia/deep_learning/code/assignment_4/video_rec")
# need to disable pylint here, as the utily can only be imported once setting directory
from utils import get_frames, store_frames  # pylint: disable=wrong-import-position,import-error


VIDEO_DIR = "/users/jmatthia/deep_learning/data/UCF50/UCF50"
FRAME_DIR = "/users/jmatthia/deep_learning/data/UCF50_frames"
SPLIT_DIR = "/users/jmatthia/deep_learning/data/UCF50_splits"
N_FRAMES = 16


@dataclass
class SplitConfig:
    """Hyperparameters for train/val/test splitting."""
    ts_ratio: float = 0.10
    val_ratio: float = 0.20
    seed: int = 42

    @property
    def relative_val(self):
        """Val fraction of the train+val pool."""
        return self.val_ratio / (1.0 - self.ts_ratio)


def extract_frames(video_dir, frame_dir, n_frames):
    """Extract n_frames uniformly from each video and write them to frame_dir."""
    for action_class in tqdm(sorted(os.listdir(video_dir)), desc="Extracting frames"):
        for video_file in sorted(os.listdir(os.path.join(video_dir, action_class))):
            video_path = os.path.join(video_dir, action_class, video_file)
            store_path = os.path.join(
                frame_dir, action_class, os.path.splitext(video_file)[0]
            )
            os.makedirs(store_path, exist_ok=True)
            frames, _ = get_frames(video_path, n_frames=n_frames)
            store_frames(frames, store_path)


def build_dataset(frame_dir):
    """Return parallel arrays of video paths, labels, and group IDs from frame_dir."""
    vid_paths, vid_labels, vid_groups = [], [], []
    for action_class in sorted(os.listdir(frame_dir)):
        class_path = os.path.join(frame_dir, action_class)
        if not os.path.isdir(class_path):
            continue
        for vid_folder in sorted(os.listdir(class_path)):
            match = re.search(r'_g(\d+)_', vid_folder)
            if match is None:
                continue
            vid_paths.append(os.path.join(class_path, vid_folder))
            vid_labels.append(action_class)
            vid_groups.append(f"{action_class}_{match.group(1)}")
    return np.array(vid_paths), np.array(vid_labels), np.array(vid_groups)


def _gss_split(arrays, groups, test_size, seed):
    """One-shot GroupShuffleSplit; returns (train_idx, test_idx) or raises ValueError."""
    return next(
        GroupShuffleSplit(1, test_size=test_size, random_state=seed)
        .split(arrays[0], groups=groups)
    )


def split_dataset(vid_paths, vid_labels, vid_groups, cfg=None):
    """Split videos into train/val/test by group to prevent leakage."""
    cfg = cfg or SplitConfig()
    tr_split, val_split, ts_split = [], [], []

    for action_class in np.unique(vid_labels):
        mask = vid_labels == action_class

        try:
            tr_val_idx, ts_idx = _gss_split(
                [vid_paths[mask]], vid_groups[mask], cfg.ts_ratio, cfg.seed
            )
        except ValueError:
            print(f"WARNING: {action_class} too few groups for test split — skipping")
            continue

        try:
            tr_idx, val_idx = _gss_split(
                [vid_paths[mask][tr_val_idx]], vid_groups[mask][tr_val_idx],
                cfg.relative_val, cfg.seed
            )
        except ValueError:
            print(f"WARNING: {action_class} too few groups for val split — skipping")
            continue

        tr_split  += list(zip(vid_paths[mask][tr_val_idx][tr_idx],
                              vid_labels[mask][tr_val_idx][tr_idx]))
        val_split += list(zip(vid_paths[mask][tr_val_idx][val_idx],
                              vid_labels[mask][tr_val_idx][val_idx]))
        ts_split  += list(zip(vid_paths[mask][ts_idx],
                              vid_labels[mask][ts_idx]))

    return tr_split, val_split, ts_split


def copy_splits(split_dir, tr_split, val_split, ts_split):
    """Copy frame folders into train/val/test directory trees under split_dir."""
    for split_name, split_data in [("train", tr_split), ("val", val_split), ("test", ts_split)]:
        for vid_path, label in tqdm(split_data, desc=split_name):
            dst = os.path.join(split_dir, split_name, label, os.path.basename(vid_path))
            shutil.copytree(str(vid_path), dst, dirs_exist_ok=True)


def check_leakage(split_dir):
    """Assert no group overlap across splits and that all 50 classes are present."""
    def get_groups(split_name):
        """Collect group IDs for all videos in a split."""
        groups = set()
        split_path = os.path.join(split_dir, split_name)
        for cls in os.listdir(split_path):
            for vid in os.listdir(os.path.join(split_path, cls)):
                match = re.search(r'_g(\d+)_', vid)
                if match:
                    groups.add(f"{cls}_{match.group(1)}")
        return groups

    tr_g, val_g, ts_g = get_groups("train"), get_groups("val"), get_groups("test")
    print(f"Train groups: {len(tr_g)} | Val groups: {len(val_g)} | Test groups: {len(ts_g)}")

    assert len(tr_g & val_g) == 0, f"LEAKAGE: train/val share {len(tr_g & val_g)} groups"
    assert len(tr_g & ts_g)  == 0, f"LEAKAGE: train/test share {len(tr_g & ts_g)} groups"
    assert len(val_g & ts_g) == 0, f"LEAKAGE: val/test share {len(val_g & ts_g)} groups"
    print("No group leakage detected.")

    for split_name in ("train", "val", "test"):
        n_classes = len(os.listdir(os.path.join(split_dir, split_name)))
        print(f"{split_name}: {n_classes} classes")
    assert all(
        len(os.listdir(os.path.join(split_dir, s))) == 50
        for s in ("train", "val", "test")
    ), "Not all 50 classes present in every split"
    print("All 50 classes present in all splits.")


def main():
    """Run full pipeline: extract frames, build dataset, split, copy, verify."""
    extract_frames(VIDEO_DIR, FRAME_DIR, N_FRAMES)

    vid_paths, vid_labels, vid_groups = build_dataset(FRAME_DIR)
    print(f"Total videos: {len(vid_paths)}")

    tr_split, val_split, ts_split = split_dataset(vid_paths, vid_labels, vid_groups)
    print(f"Train: {len(tr_split)} | Val: {len(val_split)} | Test: {len(ts_split)}")

    copy_splits(SPLIT_DIR, tr_split, val_split, ts_split)

    check_leakage(SPLIT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
