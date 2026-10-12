"""
prepare_data.py  — Step 0 of the pipeline.

Downloads MVTec AD 'bottle' category as a proxy for lyophilized vial images,
remaps class names to pharma-equivalent labels, builds the metadata CSV,
and produces frozen train/val/test splits (vial-level, no leakage).

Run:
    uv run python src/prepare_data.py
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import snapshot_download

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = Path("data/proxy")

# Map MVTec bottle defect categories → pharma analogs
CLASS_MAP = {
    "good":            "defect_free",
    "broken_large":    "collapse",
    "broken_small":    "meltback",
    "contamination":   "skinning",
}
CLASS_NAMES = sorted(set(CLASS_MAP.values()))
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

SEED = 42


def download_bottle():
    """Download bottle category from MVTec AD mirror on Hugging Face."""
    print("Downloading MVTec bottle dataset from Hugging Face (foersben/mvtec-ad)...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="foersben/mvtec-ad",
        repo_type="dataset",
        allow_patterns="bottle/**",
        local_dir=str(DATA_DIR),
    )
    print("Download complete.")


def extract_bottle():
    """Extract bottle category into data/proxy/."""
    print(f"Extracting {BOTTLE_TAR} ...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(BOTTLE_TAR, "r:gz") as tar:
        tar.extractall(DATA_DIR)
    print("Extraction complete.")


def collect_images() -> pd.DataFrame:
    """
    Walk extracted folder structure and build metadata DataFrame.

    MVTec bottle structure:
        bottle/
          train/good/
          test/good/
          test/broken_large/
          test/broken_small/
          test/contamination/
    We flatten everything into a single image pool then re-split ourselves
    so we control the split (vial-id level).
    """
    bottle_dir = DATA_DIR / "bottle"
    if not bottle_dir.exists():
        # Some tar files extract to the current dir
        bottle_dir = DATA_DIR

    records = []
    img_id = 0
    for split_dir in sorted(bottle_dir.glob("*")):          # train / test / ground_truth
        if not split_dir.is_dir() or split_dir.name == "ground_truth":
            continue
        for class_dir in sorted(split_dir.glob("*")):
            if not class_dir.is_dir():
                continue
            mvtec_class = class_dir.name
            pharma_class = CLASS_MAP.get(mvtec_class)
            if pharma_class is None:
                continue
            for img_path in sorted(class_dir.glob("*.png")) + sorted(class_dir.glob("*.jpg")):
                mask_path = bottle_dir / "ground_truth" / mvtec_class / f"{img_path.stem}_mask.png"
                records.append({
                    "image_path": str(img_path.resolve()),
                    "mask_path":  str(mask_path.resolve()) if mask_path.exists() else "",
                    "vial_id":    img_id,          # each image treated as a distinct "vial"
                    "batch_id":   img_id // 20,    # synthetic batch grouping
                    "defect_class": pharma_class,
                    "class_idx":  CLASS_TO_IDX[pharma_class],
                    "severity":   "high" if "large" in mvtec_class else
                                  ("low" if mvtec_class == "good" else "medium"),
                    "source":     "proxy_mvtec",
                })
                img_id += 1

    df = pd.DataFrame(records)
    print(f"Collected {len(df)} images across {df['defect_class'].nunique()} classes.")
    print(df["defect_class"].value_counts().to_string())
    return df


def make_splits(df: pd.DataFrame):
    """
    Stratified train / val / test split (70 / 15 / 15).
    Since each image is its own 'vial', we do stratified split on class.
    """
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["class_idx"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["class_idx"], random_state=SEED
    )
    train_df = train_df.copy(); train_df["split"] = "train"
    val_df   = val_df.copy();   val_df["split"]   = "val"
    test_df  = test_df.copy();  test_df["split"]  = "test"

    meta = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
    return meta, train_df, val_df, test_df


def main():
    # 1. Download if not present
    if not (DATA_DIR / "bottle").exists():
        download_bottle()

    # 2. Collect
    df = collect_images()
    if df.empty:
        raise RuntimeError(
            "No images found. Check that data/proxy/bottle/ exists and contains PNG files."
        )

    # 4. Split
    meta, train_df, val_df, test_df = make_splits(df)

    # 5. Save
    Path("results/metrics").mkdir(parents=True, exist_ok=True)
    meta.to_csv("results/metrics/metadata.csv", index=False)

    # Class distribution
    dist = (
        meta.groupby(["split", "defect_class"])
        .size()
        .unstack(fill_value=0)
    )
    dist.to_csv("results/metrics/class_distribution.csv")
    print("\nClass distribution per split:")
    print(dist.to_string())

    # Leak check
    train_ids = set(train_df["vial_id"])
    test_ids  = set(test_df["vial_id"])
    overlap = train_ids & test_ids
    assert len(overlap) == 0, f"Split leak! {len(overlap)} vial IDs in both train and test."
    print("\n✓ No train/test vial-ID overlap confirmed.")
    print(f"\n✓ metadata.csv saved → results/metrics/metadata.csv")
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print("\nClass names:", CLASS_NAMES)
    print("Class→idx:", CLASS_TO_IDX)


if __name__ == "__main__":
    main()
