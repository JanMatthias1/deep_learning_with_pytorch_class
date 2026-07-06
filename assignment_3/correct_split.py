from pathlib import Path
import hashlib

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def assert_clean_yolo_split(voc_split_dir, show_examples=20):
    """
    Checks ONLY:
      1) image/label filename match within each split
      2) no filename (stem) appears in more than one split

    Does NOT check image content / hashes / near-duplicates.
    """

    voc_split_dir = Path(voc_split_dir)

    def stems(split, subdir, exts):
        p = voc_split_dir / subdir / split
        assert p.exists(), f"Missing {p}"
        return {
            f.stem
            for f in p.iterdir()
            if f.is_file() and f.suffix.lower() in exts
        }

    splits = {}
    for s in ("train", "val", "test"):
        img_stems = stems(s, "images", IMG_EXTS)
        lbl_stems = stems(s, "labels", {".txt"})

        if img_stems != lbl_stems:
            raise AssertionError(
                f"{s}: image/label mismatch. "
                f"Missing labels: {sorted(img_stems - lbl_stems)[:show_examples]}, "
                f"Missing images: {sorted(lbl_stems - img_stems)[:show_examples]}"
            )

        splits[s] = img_stems

    # ---- filename leakage check ----
    def assert_no_overlap(a, b):
        overlap = splits[a] & splits[b]
        if overlap:
            raise AssertionError(
                f"{a}/{b}: filename overlap ({len(overlap)}). "
                f"Examples: {sorted(overlap)[:show_examples]}"
            )

    assert_no_overlap("train", "val")
    assert_no_overlap("train", "test")
    assert_no_overlap("val", "test")

    print("✓ YOLO split OK (filename-level):",
          {s: len(v) for s, v in splits.items()})