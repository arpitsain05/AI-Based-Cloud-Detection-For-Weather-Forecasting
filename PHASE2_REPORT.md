# Phase 2A Report: EfficientNet-B0 Retraining on 15-Class Weather Dataset

Generated: 2026-07-18

## 📋 Training Configuration

* **Model Backbone**: `EfficientNet-B0` (Pretrained on ImageNet)
* **Custom Head**: Fully-connected classification head mapping to `15` classes.
* **Input Image Size**: `224x224` pixels
* **Optimization Setup**:
  * **Optimizer**: `AdamW` (learning rate = `1e-4`, weight decay = `1e-5`)
  * **Scheduler**: `ReduceLROnPlateau` (mode = `min`, factor = `0.5`, patience = `2`)
* **Early Stopping**: Patience = `5` epochs
* **Device**: `cuda` (NVIDIA GeForce RTX 3050 Laptop GPU)
* **AMP (Automatic Mixed Precision)**: Enabled

---

## ⏱️ Early Stopping & Epoch Progression

The model training was completed in **11 epochs** when early stopping was triggered. The validation loss and accuracy progressed as follows:

* **Epoch 1**: train_loss = `1.4165`, val_loss = `0.4486`, train_acc = `63.80%`, val_acc = `87.69%` (Saved best checkpoint)
* **Epoch 2**: train_loss = `0.5145`, val_loss = `0.2992`, train_acc = `84.47%`, val_acc = `90.72%` (Saved best checkpoint)
* **Epoch 3**: train_loss = `0.3892`, val_loss = `0.2527`, train_acc = `87.73%`, val_acc = `91.82%` (Saved best checkpoint)
* **Epoch 4**: train_loss = `0.3147`, val_loss = `0.2431`, train_acc = `89.72%`, val_acc = `92.13%` (Saved best checkpoint)
* **Epoch 5**: train_loss = `0.2690`, val_loss = `0.2295`, train_acc = `91.03%`, val_acc = `92.13%` (Saved best checkpoint)
* **Epoch 6**: train_loss = `0.2364`, val_loss = `0.2144`, train_acc = `92.42%`, val_acc = `93.53%` (Saved best checkpoint)
* **Epoch 7**: train_loss = `0.2150`, val_loss = `0.2239`, train_acc = `92.92%`, val_acc = `92.36%` 
* **Epoch 8**: train_loss = `0.1949`, val_loss = `0.2192`, train_acc = `93.90%`, val_acc = `92.60%`
* **Epoch 9**: train_loss = `0.1618`, val_loss = `0.2272`, train_acc = `94.70%`, val_acc = `92.67%`
* **Epoch 10**: train_loss = `0.1519`, val_loss = `0.2149`, train_acc = `95.11%`, val_acc = `93.22%`
* **Epoch 11**: train_loss = `0.1343`, val_loss = `0.2337`, train_acc = `95.61%`, val_acc = `92.52%` (Early stopping triggered)

### 🏆 Best Validation Metrics (Epoch 6)
* **Validation Loss**: `0.2144`
* **Validation Accuracy**: `93.53%`

---

## 🧠 Model Checkpoint Filenames

* **Best Model Weights**: `models/efficientnet_b0_15class_best.pth`
* **Last Training Checkpoint**: `models/efficientnet_b0_15class_last.pth`
* *Note: The old 5-class model files (`best_model.pth` and `last_checkpoint.pth`) were preserved.*

---

## 📈 Training Graphs Generated

Saved in `outputs/training/`:
* `loss_curve.png`: Shows train vs validation loss convergence.
* `accuracy_curve.png`: Shows train vs validation accuracy metrics.

---

## 🧪 Test Set Evaluation Metrics (1,305 test images)

Evaluated using the best model weights checkpoint (`models/efficientnet_b0_15class_best.pth`):

| Metric | Weighted Average Value |
| :--- | :---: |
| **Test Loss** | `0.2267` |
| **Accuracy** | `93.79%` |
| **Precision** | `93.86%` |
| **Recall** | `93.79%` |
| **F1-Score** | `93.79%` |

---

## 📊 Classification Report Summary

The class-wise evaluation details are summarized below:

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **cloudy** | `0.935` | `0.983` | `0.959` | 59 |
| **cyclone** | `1.000` | `1.000` | `1.000` | 22 |
| **dew** | `0.980` | `0.943` | `0.962` | 106 |
| **foggy** | `0.953` | `0.989` | `0.971` | 186 |
| **frost** | `0.892` | `0.806` | `0.847` | 72 |
| **glaze** | `0.835` | `0.835` | `0.835` | 97 |
| **hail** | `0.978` | `0.978` | `0.978` | 90 |
| **lightning** | `1.000` | `1.000` | `1.000` | 58 |
| **rainbow** | `1.000` | `0.972` | `0.986` | 36 |
| **rainy** | `0.972` | `0.920` | `0.945` | 112 |
| **rime** | `0.890` | `0.925` | `0.907` | 174 |
| **sandstorm** | `0.990` | `0.971` | `0.981` | 105 |
| **shine** | `0.973` | `0.923` | `0.947` | 39 |
| **snow** | `0.847` | `0.883` | `0.865` | 94 |
| **sunrise** | `0.982` | `1.000` | `0.991` | 55 |

---

## 🎯 Confusion Matrix

* Saved to: `outputs/training/confusion_matrix.png`
* High confusion was primarily observed between ice-related classes like **rime**, **frost**, and **glaze**, which is typical given their extremely similar texture patterns in visual imagery. Distinct phenomena like **cyclone** and **lightning** were classified with **100% precision and recall**.

---

## 🏁 Final Conclusion

The retraining of the `EfficientNet-B0` model on the expanded 15-class dataset was highly successful. The model achieved a **93.79% test accuracy**, representing extremely strong feature extraction capabilities. Generalization is excellent, particularly on complex visual conditions. The model checkpoints have been correctly saved in `models/` under the new naming convention, ensuring backwards compatibility with the original 5-class checkpoint.
