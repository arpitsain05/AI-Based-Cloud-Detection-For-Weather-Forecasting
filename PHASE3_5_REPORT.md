# Phase 3.5 Report

Generated: 2026-07-08

## Scope

Phase 3.5 performed full model training and evaluation for the AI-Based Weather Detection using Deep Learning project.

Phase 4 has not been started.

## Training Command

```powershell
.\venv\Scripts\python.exe src\train.py
```

The command used the existing project configuration:

- Backbone: EfficientNet-B0
- Pretrained weights: ImageNet
- Loss: CrossEntropyLoss
- Optimizer: AdamW
- Scheduler: ReduceLROnPlateau
- Early stopping patience: 5
- Device: CPU
- AMP: Disabled because CUDA was unavailable
- Epoch limit: 15 from `src/config.py`
- Batch size: 32 from `src/config.py`
- Learning rate: 0.0001 from `src/config.py`
- Num workers: 0 from `src/config.py`

## Dataset

| split | images |
| --- | --- |
| train | 909 |
| validation | 193 |
| test | 201 |

Classes:

- cloudy
- cyclone
- rainy
- shine
- sunrise

## Training Summary

- Training completed successfully.
- Early stopping triggered after epoch 14.
- Best validation accuracy first occurred at epoch 8.
- Best validation accuracy: 0.9689
- Total training time: approximately 1319.6 seconds, about 22 minutes
- Best model saved to: `models/best_model.pth`
- Last checkpoint saved to: `models/last_checkpoint.pth`

## Test Evaluation

The best model checkpoint was evaluated on the test dataset.

| metric | value |
| --- | --- |
| Test loss | 0.032364993166196995 |
| Final accuracy | 0.9900497512437811 |
| Precision | 0.9905351292318894 |
| Recall | 0.9900497512437811 |
| F1 score | 0.9901266783531701 |

## Confusion Matrix

Rows are true labels and columns are predicted labels in this order:

`cloudy`, `cyclone`, `rainy`, `shine`, `sunrise`

| class | cloudy | cyclone | rainy | shine | sunrise |
| --- | --- | --- | --- | --- | --- |
| cloudy | 50 | 0 | 0 | 1 | 0 |
| cyclone | 0 | 23 | 0 | 0 | 0 |
| rainy | 0 | 0 | 33 | 0 | 0 |
| shine | 0 | 0 | 0 | 39 | 0 |
| sunrise | 0 | 0 | 0 | 1 | 54 |

## Generated Artifacts

Model artifacts:

- `models/best_model.pth`
- `models/last_checkpoint.pth`

Training/evaluation artifacts:

- `outputs/training/metrics.json`
- `outputs/training/metrics.csv`
- `outputs/training/training_history.json`
- `outputs/training/training_history.csv`
- `outputs/training/loss_curve.png`
- `outputs/training/accuracy_curve.png`
- `outputs/training/confusion_matrix.png`

## Warnings and Notes

- CUDA was unavailable, so training ran on CPU and AMP was disabled.
- TensorBoard package was not installed in the active virtual environment, so TensorBoard event files were not generated during this run.
- Albumentations reported that `ShiftScaleRotate` is a special case of `Affine`; this warning is non-blocking and comes from the existing transform pipeline.

## Status

Phase 3.5 training and best-checkpoint evaluation are complete. Phase 4 has not been started.
