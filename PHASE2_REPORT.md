# Phase 2 Report

Generated: 2026-07-08 11:56:45

## Scope

Phase 2 analyzed the real dataset, verified the existing processed split, saved EDA artifacts, checked corrupted/unreadable images without deleting files, and validated the PyTorch DataLoader/transform pipeline.

## Raw Dataset

- Raw data directory: `data\raw`
- Detected classes: `cloudy, cyclone, rainy, shine, sunrise`
- Total valid images: `1305`
- Corrupted/unreadable images reported: `0`
- Skipped non-image files: `1`

| class | valid_images |
| --- | --- |
| cloudy | 336 |
| cyclone | 144 |
| rainy | 215 |
| shine | 253 |
| sunrise | 357 |

## Processed Split Verification

- Processed data directory: `data\processed`
- Existing split reused: `True`
- Existing split valid: `True`
- Split was not recreated or overwritten.

| class | raw_total | train | val | test | is_valid |
| --- | --- | --- | --- | --- | --- |
| cloudy | 336 | 235 | 50 | 51 | True |
| cyclone | 144 | 100 | 21 | 23 | True |
| rainy | 215 | 150 | 32 | 33 | True |
| shine | 253 | 177 | 37 | 39 | True |
| sunrise | 357 | 249 | 53 | 55 | True |

## Transform and DataLoader Verification

- Image size: `224x224`
- Batch size: `32`
- Config normalization mean: `[0.4611, 0.4577, 0.4499]`
- Config normalization std: `[0.2748, 0.2562, 0.2911]`
- Recomputed train mean: `[0.4611, 0.4577, 0.4499]`
- Recomputed train std: `[0.2748, 0.2562, 0.2911]`
- Pipeline verified: `True`

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
