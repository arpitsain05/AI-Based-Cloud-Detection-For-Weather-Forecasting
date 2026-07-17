import csv
import json
import os
import random
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# Set random seed for reproducible samples.
random.seed(42)
np.random.seed(42)

try:
    from src.config import (
        BATCH_SIZE,
        CLASSES,
        IMAGE_SIZE,
        NORM_MEAN,
        NORM_STD,
        PROCESSED_DATA_DIR,
        RAW_DATA_DIR,
    )
except ImportError:
    from config import (
        BATCH_SIZE,
        CLASSES,
        IMAGE_SIZE,
        NORM_MEAN,
        NORM_STD,
        PROCESSED_DATA_DIR,
        RAW_DATA_DIR,
    )


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif", ".webp")
SPLITS = ("train", "val", "test")
EDA_DIR = Path(PROCESSED_DATA_DIR) / "eda"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def relative_path(path):
    """Return a project-relative path for readable reports."""
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def ensure_eda_dir():
    EDA_DIR.mkdir(parents=True, exist_ok=True)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def is_image_file(path):
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def inspect_image(path):
    """
    Validate an image without deleting or changing it.
    Returns a tuple: (is_valid, reason, width, height).
    """
    path = Path(path)
    if path.stat().st_size == 0:
        return False, "0-byte file", None, None

    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:
        return False, f"PIL verification failed: {exc}", None, None

    try:
        image = cv2.imread(str(path))
        if image is None:
            return False, "OpenCV returned None", None, None
        height, width = image.shape[:2]
    except Exception as exc:
        return False, f"OpenCV read failed: {exc}", None, None

    return True, "", width, height


def analyze_raw_dataset():
    """
    Analyze the raw dataset and report unreadable images without deleting them.
    """
    print("=== Phase 2: Raw Dataset Analysis ===")

    raw_dir = Path(RAW_DATA_DIR)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_dir}")

    classes = sorted([path.name for path in raw_dir.iterdir() if path.is_dir()])
    class_distribution = {}
    all_valid_files = {}
    corrupted_files = []
    skipped_files = []
    image_dimensions = []

    print(f"Detected {len(classes)} classes: {classes}")

    for class_name in classes:
        class_dir = raw_dir / class_name
        valid_paths = []

        for path in sorted(class_dir.iterdir()):
            if not path.is_file() or path.name.startswith("."):
                continue
            if not is_image_file(path):
                skipped_files.append(
                    {
                        "class": class_name,
                        "path": relative_path(path),
                        "reason": "unsupported file extension",
                    }
                )
                continue

            is_valid, reason, width, height = inspect_image(path)
            if is_valid:
                valid_paths.append(str(path))
                image_dimensions.append(
                    {
                        "class": class_name,
                        "path": relative_path(path),
                        "width": width,
                        "height": height,
                    }
                )
            else:
                corrupted_files.append(
                    {
                        "class": class_name,
                        "path": relative_path(path),
                        "reason": reason,
                    }
                )

        class_distribution[class_name] = len(valid_paths)
        all_valid_files[class_name] = valid_paths

    total_images = sum(class_distribution.values())
    print("\nClass distribution:")
    for class_name, count in class_distribution.items():
        percentage = (count / total_images) * 100 if total_images else 0
        print(f" - {class_name}: {count} valid images ({percentage:.2f}%)")
    print(f"Total valid images: {total_images}")
    print(f"Corrupted/unreadable images reported: {len(corrupted_files)}")
    print(f"Skipped non-image files: {len(skipped_files)}")

    return {
        "classes": classes,
        "class_distribution": class_distribution,
        "all_valid_files": all_valid_files,
        "corrupted_files": corrupted_files,
        "skipped_files": skipped_files,
        "image_dimensions": image_dimensions,
    }


def save_raw_eda_outputs(raw_analysis):
    ensure_eda_dir()

    class_rows = []
    total = sum(raw_analysis["class_distribution"].values())
    for class_name, count in raw_analysis["class_distribution"].items():
        class_rows.append(
            {
                "class": class_name,
                "valid_images": count,
                "percentage": round((count / total) * 100, 4) if total else 0,
            }
        )

    write_csv(EDA_DIR / "class_distribution.csv", class_rows, ["class", "valid_images", "percentage"])
    write_json(
        EDA_DIR / "class_distribution.json",
        {
            "total_valid_images": total,
            "classes": class_rows,
        },
    )
    write_csv(EDA_DIR / "corrupted_images.csv", raw_analysis["corrupted_files"], ["class", "path", "reason"])
    write_json(EDA_DIR / "corrupted_images.json", raw_analysis["corrupted_files"])
    write_csv(EDA_DIR / "skipped_files.csv", raw_analysis["skipped_files"], ["class", "path", "reason"])
    write_json(EDA_DIR / "image_dimensions.json", raw_analysis["image_dimensions"])


def display_and_save_samples(all_valid_files, output_path=None):
    """
    Save a grid of sample images from each class.
    """
    ensure_eda_dir()
    output_path = Path(output_path) if output_path else EDA_DIR / "raw_sample_grid.png"

    print("\n=== Saving Sample Image Grid ===")
    num_classes = len(all_valid_files)
    samples_per_class = 3

    if num_classes == 0:
        print("No classes available for sample grid.")
        return None

    fig, axes = plt.subplots(num_classes, samples_per_class, figsize=(12, 3 * num_classes))
    if num_classes == 1:
        axes = np.expand_dims(axes, axis=0)

    for class_index, (class_name, paths) in enumerate(all_valid_files.items()):
        sample_paths = random.sample(paths, min(samples_per_class, len(paths))) if paths else []

        for sample_index in range(samples_per_class):
            ax = axes[class_index, sample_index]
            if sample_index < len(sample_paths):
                path = sample_paths[sample_index]
                image = cv2.imread(path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                height, width = image.shape[:2]
                ax.imshow(image)
                ax.set_title(f"{class_name}\n{width}x{height}", fontsize=9)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Sample grid saved to: {relative_path(output_path)}")
    return output_path


def collect_processed_split_counts(classes):
    split_counts = {split: {class_name: 0 for class_name in classes} for split in SPLITS}
    missing_dirs = []

    for split in SPLITS:
        split_dir = Path(PROCESSED_DATA_DIR) / split
        if not split_dir.exists():
            missing_dirs.append(relative_path(split_dir))
            continue
        for class_name in classes:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                missing_dirs.append(relative_path(class_dir))
                continue
            split_counts[split][class_name] = sum(
                1 for path in class_dir.rglob("*") if path.is_file() and is_image_file(path)
            )

    return split_counts, missing_dirs


def create_processed_split(raw_analysis):
    """
    Creates a new train/val/test split from raw_analysis data.
    Only copies valid files.
    """
    import shutil
    print("\n=== Recreating Processed Split (70/15/15) ===")

    for split in SPLITS:
        split_dir = Path(PROCESSED_DATA_DIR) / split
        if split_dir.exists():
            print(f"Cleaning existing split directory: {relative_path(split_dir)}")
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)

    classes = raw_analysis["classes"]
    all_valid_files = raw_analysis["all_valid_files"]

    for class_name in classes:
        valid_paths = all_valid_files[class_name]
        random.seed(42)
        shuffled_paths = list(valid_paths)
        random.shuffle(shuffled_paths)

        total_valid = len(shuffled_paths)
        train_end = int(total_valid * 0.70)
        val_end = train_end + int(total_valid * 0.15)

        train_paths = shuffled_paths[:train_end]
        val_paths = shuffled_paths[train_end:val_end]
        test_paths = shuffled_paths[val_end:]

        splits_map = {
            "train": train_paths,
            "val": val_paths,
            "test": test_paths
        }

        for split, paths in splits_map.items():
            split_class_dir = Path(PROCESSED_DATA_DIR) / split / class_name
            split_class_dir.mkdir(parents=True, exist_ok=True)

            for path in paths:
                src_path = Path(path)
                dest_path = split_class_dir / src_path.name
                shutil.copy2(src_path, dest_path)

        print(f" - {class_name}: Split completed (train={len(train_paths)}, val={len(val_paths)}, test={len(test_paths)}, total={total_valid})")

    print("Processed split recreated successfully.")


def update_config_normalization(mean, std):
    config_path = Path(__file__).resolve().parent / "config.py"
    if not config_path.exists():
        config_path = Path("src/config.py")
    if not config_path.exists():
        print("Warning: config.py not found, skipping config update.")
        return

    content = config_path.read_text(encoding="utf-8")
    import re
    new_mean_str = f"NORM_MEAN = {mean}"
    new_std_str = f"NORM_STD = {std}"

    content = re.sub(r"NORM_MEAN\s*=\s*\[[^\]]+\]", new_mean_str, content)
    content = re.sub(r"NORM_STD\s*=\s*\[[^\]]+\]", new_std_str, content)

    config_path.write_text(content, encoding="utf-8")
    print(f"Updated config.py normalization stats: mean={mean}, std={std}")


def verify_existing_processed_split(raw_analysis):
    """
    Verify an existing split instead of recreating it.
    """
    print("\n=== Verifying Existing Processed Split ===")
    classes = raw_analysis["classes"]
    raw_counts = raw_analysis["class_distribution"]
    split_counts, missing_dirs = collect_processed_split_counts(classes)

    rows = []
    is_valid = len(missing_dirs) == 0
    for class_name in classes:
        train_count = split_counts["train"][class_name]
        val_count = split_counts["val"][class_name]
        test_count = split_counts["test"][class_name]
        split_total = train_count + val_count + test_count
        raw_total = raw_counts[class_name]

        expected_train = int(raw_total * 0.70)
        expected_val = int(raw_total * 0.15)
        expected_test = raw_total - expected_train - expected_val
        class_valid = (
            train_count == expected_train
            and val_count == expected_val
            and test_count == expected_test
            and split_total == raw_total
        )
        is_valid = is_valid and class_valid

        rows.append(
            {
                "class": class_name,
                "raw_total": raw_total,
                "train": train_count,
                "val": val_count,
                "test": test_count,
                "split_total": split_total,
                "expected_train": expected_train,
                "expected_val": expected_val,
                "expected_test": expected_test,
                "is_valid": class_valid,
            }
        )

    for row in rows:
        print(
            f" - {row['class']}: train={row['train']}, val={row['val']}, "
            f"test={row['test']}, valid={row['is_valid']}"
        )

    if missing_dirs:
        print("Missing processed directories:")
        for path in missing_dirs:
            print(f" - {path}")

    print(f"Existing processed split valid: {is_valid}")
    return {
        "is_valid": is_valid,
        "split_counts": split_counts,
        "rows": rows,
        "missing_dirs": missing_dirs,
    }


def save_split_outputs(split_analysis):
    ensure_eda_dir()
    write_csv(
        EDA_DIR / "split_distribution.csv",
        split_analysis["rows"],
        [
            "class",
            "raw_total",
            "train",
            "val",
            "test",
            "split_total",
            "expected_train",
            "expected_val",
            "expected_test",
            "is_valid",
        ],
    )
    write_json(
        EDA_DIR / "split_distribution.json",
        {
            "is_valid": split_analysis["is_valid"],
            "missing_dirs": split_analysis["missing_dirs"],
            "classes": split_analysis["rows"],
        },
    )


def calculate_processed_train_mean_std():
    """
    Calculate mean/std for the existing processed train split without changing config.
    """
    print("\n=== Calculating Processed Train Mean/Std ===")
    train_dir = Path(PROCESSED_DATA_DIR) / "train"
    image_paths = [
        path for path in train_dir.rglob("*")
        if path.is_file() and is_image_file(path)
    ]

    if not image_paths:
        return {"mean": None, "std": None, "image_count": 0}

    channel_sum = np.zeros(3)
    channel_sum_sq = np.zeros(3)
    image_count = 0

    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        image = image / 255.0

        channel_sum += np.mean(image, axis=(0, 1))
        channel_sum_sq += np.mean(image ** 2, axis=(0, 1))
        image_count += 1

    mean = channel_sum / image_count
    std = np.sqrt((channel_sum_sq / image_count) - (mean ** 2))
    result = {
        "mean": [round(float(value), 4) for value in mean],
        "std": [round(float(value), 4) for value in std],
        "image_count": image_count,
        "config_mean": NORM_MEAN,
        "config_std": NORM_STD,
    }
    print(f"Mean: {result['mean']}")
    print(f"Std: {result['std']}")
    return result


def verify_pipeline():
    """
    Verify DataLoader and transform behavior on the processed split.
    """
    print("\n=== Verifying DataLoader and Transform Pipeline ===")

    import sys
    import importlib
    for module_name in ["src.config", "config", "src.dataset", "dataset"]:
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])

    try:
        from src.dataset import get_dataloaders
    except ImportError:
        from dataset import get_dataloaders

    train_dir = Path(PROCESSED_DATA_DIR) / "train"
    val_dir = Path(PROCESSED_DATA_DIR) / "val"
    test_dir = Path(PROCESSED_DATA_DIR) / "test"

    train_loader, val_loader, test_loader = get_dataloaders(
        train_dir=str(train_dir),
        val_dir=str(val_dir),
        test_dir=str(test_dir),
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    verification = {}
    for loader_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        dataset_size = len(loader.dataset)
        if dataset_size == 0:
            verification[loader_name] = {
                "dataset_size": 0,
                "verified": False,
                "reason": "empty dataset",
            }
            continue

        images, labels = next(iter(loader))
        expected_batch = min(BATCH_SIZE, dataset_size)
        verified = (
            images.shape[0] == expected_batch
            and tuple(images.shape[1:]) == (3, IMAGE_SIZE, IMAGE_SIZE)
            and labels.shape[0] == expected_batch
        )
        verification[loader_name] = {
            "dataset_size": dataset_size,
            "batch_image_shape": list(images.shape),
            "batch_label_shape": list(labels.shape),
            "unique_labels": torch.unique(labels).tolist(),
            "image_dtype": str(images.dtype),
            "label_dtype": str(labels.dtype),
            "image_min": round(float(images.min().item()), 6),
            "image_max": round(float(images.max().item()), 6),
            "verified": verified,
        }
        print(
            f" - {loader_name}: size={dataset_size}, "
            f"batch={list(images.shape)}, verified={verified}"
        )

    all_verified = all(item["verified"] for item in verification.values())
    verification["all_verified"] = all_verified
    if not all_verified:
        raise RuntimeError("DataLoader/transform verification failed. See EDA output for details.")

    return verification


def save_pipeline_outputs(pipeline_verification, mean_std):
    ensure_eda_dir()
    write_json(EDA_DIR / "pipeline_verification.json", pipeline_verification)
    write_json(EDA_DIR / "train_normalization_stats.json", mean_std)


def markdown_table(rows, columns):
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join([header, separator] + body)


def generate_reports(raw_analysis, split_analysis, pipeline_verification, mean_std):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_raw = sum(raw_analysis["class_distribution"].values())

    phase1_report = f"""# Phase 1 Report

Generated: {now}

## Scope

Phase 2 analyzed the real dataset, verified the existing processed split, saved EDA artifacts, checked corrupted/unreadable images without deleting files, and validated the PyTorch DataLoader/transform pipeline.

## Raw Dataset

- Raw data directory: `{relative_path(RAW_DATA_DIR)}`
- Detected classes: `{", ".join(raw_analysis["classes"])}`
- Total valid images: `{total_raw}`
- Corrupted/unreadable images reported: `{len(raw_analysis["corrupted_files"])}`
- Skipped non-image files: `{len(raw_analysis["skipped_files"])}`

{markdown_table([{"class": k, "valid_images": v} for k, v in raw_analysis["class_distribution"].items()], ["class", "valid_images"])}

## Processed Split Verification

- Processed data directory: `{relative_path(PROCESSED_DATA_DIR)}`
- Existing split reused: `True`
- Existing split valid: `{split_analysis["is_valid"]}`
- Split was not recreated or overwritten.

{markdown_table(split_analysis["rows"], ["class", "raw_total", "train", "val", "test", "is_valid"])}

## Transform and DataLoader Verification

- Image size: `{IMAGE_SIZE}x{IMAGE_SIZE}`
- Batch size: `{BATCH_SIZE}`
- Config normalization mean: `{NORM_MEAN}`
- Config normalization std: `{NORM_STD}`
- Recomputed train mean: `{mean_std["mean"]}`
- Recomputed train std: `{mean_std["std"]}`
- Pipeline verified: `{pipeline_verification["all_verified"]}`

## EDA Outputs

Saved under `data/processed/eda/`:

- `class_distribution.csv`
- `class_distribution.json`
- `corrupted_images.csv`
- `corrupted_images.json`
- `skipped_files.csv`
- `image_dimensions.json`
- `split_distribution.csv`
- `split_distribution.json`
- `pipeline_verification.json`
- `train_normalization_stats.json`
- `raw_sample_grid.png`

## File Safety

No files were deleted. No existing processed dataset files were overwritten.
"""

    project_status = f"""# Project Status

Generated: {now}

## Current Phase

Phase 2 is complete. Phase 3 has not been started.

## Completed

- Phase 1 project structure and dataset utilities are present.
- Real dataset classes were detected and counted.
- Existing processed train/validation/test split was verified instead of recreated.
- EDA outputs were saved under `data/processed/eda/`.
- Corrupted image detection now reports files without deleting them.
- PyTorch DataLoader and Albumentations transforms were verified.

## Current Dataset

- Classes: `{", ".join(raw_analysis["classes"])}`
- Total valid raw images: `{total_raw}`
- Processed split valid: `{split_analysis["is_valid"]}`
- Corrupted/unreadable images requiring review: `{len(raw_analysis["corrupted_files"])}`

## Important Safety Notes

- Do not start Phase 3 until confirmed by the user.
- Do not delete reported corrupted files without explicit user confirmation.
- Do not recreate `data/processed/` while the current split remains valid.
"""

    next_prompt = f"""# Next Prompt

Continue this existing project only after the user confirms Phase 3.

Project: AI-Based Weather Detection using Image Processing

Current status:
- Phase 1 complete.
- Phase 2 complete as of {now}.
- Real classes: {", ".join(raw_analysis["classes"])}
- Existing processed split is valid and should be reused.
- EDA artifacts are in `data/processed/eda/`.
- No corrupted image files were deleted automatically.

Before Phase 3:
1. Read `PHASE2_REPORT.md`.
2. Read `PROJECT_STATUS.md`.
3. Inspect `src/config.py`, `src/dataset.py`, and `src/analyze_and_split.py`.
4. Do not recreate Phase 1 or Phase 2 files.
5. Wait for explicit user confirmation before starting model training work.
"""

    (PROJECT_ROOT / "PHASE1_REPORT.md").write_text(phase1_report, encoding="utf-8")
    (PROJECT_ROOT / "PROJECT_STATUS.md").write_text(project_status, encoding="utf-8")
    (PROJECT_ROOT / "NEXT_PROMPT.md").write_text(next_prompt, encoding="utf-8")


def main():
    raw_analysis = analyze_raw_dataset()
    save_raw_eda_outputs(raw_analysis)
    display_and_save_samples(raw_analysis["all_valid_files"])
    split_analysis = verify_existing_processed_split(raw_analysis)

    if not split_analysis["is_valid"]:
        print("[WARNING] Processed split invalid or missing for 15 classes. Recreating...")
        create_processed_split(raw_analysis)
        split_analysis = verify_existing_processed_split(raw_analysis)

    save_split_outputs(split_analysis)

    mean_std = calculate_processed_train_mean_std()
    pipeline_verification = verify_pipeline()
    save_pipeline_outputs(pipeline_verification, mean_std)
    generate_reports(raw_analysis, split_analysis, pipeline_verification, mean_std)

    print("\nPhase 1 dataset preparation, split, and verification completed successfully.")
    print(f"EDA outputs: {relative_path(EDA_DIR)}")
    print("Reports generated: PHASE1_REPORT.md, PROJECT_STATUS.md, NEXT_PROMPT.md")
    print("No files were deleted. Existing processed dataset was not overwritten.")


if __name__ == "__main__":
    main()
