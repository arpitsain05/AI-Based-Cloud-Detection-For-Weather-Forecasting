# Phase 1 Report

Generated: 2026-07-17 18:06:28

## Scope

Phase 2 analyzed the real dataset, verified the existing processed split, saved EDA artifacts, checked corrupted/unreadable images without deleting files, and validated the PyTorch DataLoader/transform pipeline.

## Raw Dataset

- Raw data directory: `data\raw`
- Detected classes: `cloudy, cyclone, dew, foggy, frost, glaze, hail, lightning, rainbow, rainy, rime, sandstorm, shine, snow, sunrise`
- Total valid images: `8604`
- Corrupted/unreadable images reported: `0`
- Skipped non-image files: `1`

| class | valid_images |
| --- | --- |
| cloudy | 389 |
| cyclone | 144 |
| dew | 698 |
| foggy | 1235 |
| frost | 475 |
| glaze | 639 |
| hail | 591 |
| lightning | 377 |
| rainbow | 232 |
| rainy | 741 |
| rime | 1160 |
| sandstorm | 692 |
| shine | 253 |
| snow | 621 |
| sunrise | 357 |

## Processed Split Verification

- Processed data directory: `data\processed`
- Existing split reused: `True`
- Existing split valid: `True`
- Split was not recreated or overwritten.

| class | raw_total | train | val | test | is_valid |
| --- | --- | --- | --- | --- | --- |
| cloudy | 389 | 272 | 58 | 59 | True |
| cyclone | 144 | 100 | 21 | 23 | True |
| dew | 698 | 488 | 104 | 106 | True |
| foggy | 1235 | 864 | 185 | 186 | True |
| frost | 475 | 332 | 71 | 72 | True |
| glaze | 639 | 447 | 95 | 97 | True |
| hail | 591 | 413 | 88 | 90 | True |
| lightning | 377 | 263 | 56 | 58 | True |
| rainbow | 232 | 162 | 34 | 36 | True |
| rainy | 741 | 518 | 111 | 112 | True |
| rime | 1160 | 812 | 174 | 174 | True |
| sandstorm | 692 | 484 | 103 | 105 | True |
| shine | 253 | 177 | 37 | 39 | True |
| snow | 621 | 434 | 93 | 94 | True |
| sunrise | 357 | 249 | 53 | 55 | True |

## Transform and DataLoader Verification

- Image size: `224x224`
- Batch size: `32`
- Config normalization mean: `[0.5078, 0.5134, 0.4963]`
- Config normalization std: `[0.2585, 0.2463, 0.2799]`
- Recomputed train mean: `[0.5078, 0.5134, 0.4963]`
- Recomputed train std: `[0.2585, 0.2463, 0.2799]`
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
