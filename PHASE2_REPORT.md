# Phase 2 Report: Model Training & Evaluation (15-Class Weather Dataset)

Generated: 2026-07-18

This phase documents the training and evaluation of two deep learning architectures on the expanded 15-class weather dataset.

---

## 🟢 Phase 2A: EfficientNet-B0 Retraining

### 📋 Training Configuration
* **Model Backbone**: `EfficientNet-B0` (Pretrained on ImageNet)
* **Classifier Head**: Fully-connected layer mapping to `15` classes.
* **Input Image Size**: `224x224` pixels
* **Optimizer**: `AdamW` (learning rate = `1e-4`, weight decay = `1e-5`)
* **Scheduler**: `ReduceLROnPlateau` (mode = `min`, factor = `0.5`, patience = `2`)
* **Early Stopping**: Patience = `5` epochs
* **Device**: `cuda` (NVIDIA GeForce RTX 3050 Laptop GPU)
* **AMP (Automatic Mixed Precision)**: Enabled

### ⏱️ Training Progression
Training ran for **11 epochs** before early stopping was triggered. The best checkpoint was saved at **Epoch 6**:
* **Best Validation Loss**: `0.2144`
* **Best Validation Accuracy**: `93.53%`

### 🧠 Model Checkpoints
* **Best Checkpoint**: `models/efficientnet_b0_15class_best.pth`
* **Last Checkpoint**: `models/efficientnet_b0_15class_last.pth`

---

## 🔵 Phase 2B: Swin Transformer Training

### 📋 Training Configuration
* **Model Backbone**: `Swin Transformer (swin_t)` (Pretrained on ImageNet)
* **Classifier Head**: Replaced `model.head` with Dropout (`0.2`) and Linear mapping to `15` classes.
* **Input Image Size**: `224x224` pixels
* **Optimizer**: `AdamW` (learning rate = `1e-4`, weight decay = `1e-5`)
* **Scheduler**: `ReduceLROnPlateau` (mode = `min`, factor = `0.5`, patience = `2`)
* **Early Stopping**: Patience = `5` epochs
* **Device**: `cuda` (NVIDIA GeForce RTX 3050 Laptop GPU)
* **AMP (Automatic Mixed Precision)**: Enabled

### ⏱️ Training Progression
Training ran for **10 epochs** before early stopping was triggered. The best checkpoint was saved at **Epoch 5**:
* **Best Validation Loss**: `0.1962`
* **Best Validation Accuracy**: `93.61%`

### 🧠 Model Checkpoints
* **Best Checkpoint**: `models/swin_transformer_15class_best.pth`
* **Last Checkpoint**: `models/swin_transformer_15class_last.pth`

---

## 📊 Comprehensive Model Comparison

Both models were evaluated on the independent **test split** consisting of 1,305 images:

| Metric / Dimension | 🟢 EfficientNet-B0 (Baseline) | 🔵 Swin Transformer | Difference |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **`93.79%`** | `93.56%` | `-0.23%` |
| **Precision (Weighted)** | **`93.86%`** | `93.64%` | `-0.22%` |
| **Recall (Weighted)** | **`93.79%`** | `93.56%` | `-0.23%` |
| **F1 Score (Weighted)** | **`93.79%`** | `93.58%` | `-0.21%` |
| **Test Loss** | **`0.2267`** | `0.2332` | `+0.0065` |
| **Parameters (Total/Trainable)**| **`4,026,763`** | `27,530,889` | ~6.8x larger |
| **Model Weight Size** | **`46.56 MB`** | `315.51 MB` | ~6.8x larger |
| **Inference Time (GPU)** | **`17.519 ms/image`** | `25.152 ms/image` | +43.5% slower |
| **Training Time (GPU)** | **`~348 seconds (5.8m)`** | `1,213 seconds (20.2m)`| ~3.5x slower |

---

## 🎯 Confusion Matrix & Error Analysis

* **EfficientNet-B0 Matrix**: Saved at `outputs/training/confusion_matrix.png`
* **Swin Transformer Matrix**: Saved at `outputs/training/swin/confusion_matrix.png`
* **Analysis**:
  * Distinct meteorological phenomena like **cyclone** and **lightning** were classified with **100% precision and recall** by both models.
  * Ice-related and snow-related classes (**rime**, **frost**, **glaze**, and **snow**) showed some mutual confusion due to highly similar visual textures. 
  * Swin Transformer slightly improved classification accuracy on ice-textures (e.g., +1.04% F1-score on **frost** and +0.75% on **glaze**), but lost marginal ground on precipitation categories (e.g., -2.61% F1-score on **rainy**).

---

## 🏁 Final Conclusion & Model Selection

**EfficientNet-B0 is selected as the primary deployment model** for the weather forecasting system.

### Rationale:
1. **Performance**: EfficientNet-B0 achieves slightly better overall classification accuracy (`93.79%` vs `93.56%`).
2. **Efficiency**:
   * **Size**: It is significantly smaller (`46.56 MB` vs `315.51 MB`), which translates to lower deployment costs, smaller container sizes, and lower memory overhead.
   * **Inference Speed**: It runs **43.5% faster** during inference (`17.52 ms` vs `25.15 ms`), making it much better suited for real-time applications and low-resource environments.
   * **Training Speed**: It trains **3.5x faster**, facilitating easier model retraining and faster experiment cycles.
