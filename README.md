# AI-Based Cloud Detection & Weather Forecasting (IMD)

An advanced Deep Learning project designed for the **Indian Meteorological Department (IMD)** to classify cloud and satellite imagery into five distinct meteorological categories. The system utilizes a state-of-the-art **EfficientNet-B0** convolutional neural network to achieve near-perfect classification performance and features an interactive, beautiful Streamlit web application with built-in **Grad-CAM** explainability.

---

## 📋 Project Overview

Weather satellite images can be challenging to classify manually due to variations in lighting, sun glint, haze, and complex cloud formations. This project delivers a high-accuracy, end-to-end automated pipeline to preprocess weather scene images, train a robust classifier, run single-image inference, and present predictions in a custom-designed web dashboard. 

The dashboard provides real-time predictions, per-class confidence distributions, and **Grad-CAM** attention visualization. This ensures that meteorologists do not just see the *what* (the predicted category) but also the *why* (the specific visual features, such as cyclonic bands or rainy clouds, that influenced the model).

---

## 🚀 Features

*   **State-of-the-Art Classifier**: Employs transfer learning on `EfficientNet-B0` with a customized fully-connected classification head.
*   **Robust Preprocessing & Augmentation**: Utilizes the high-performance `Albumentations` library to apply specialized spatial transforms (horizontal/vertical flips, 90-degree rotations, shift-scale-rotate) and pixel-level adjustments.
*   **Explainable AI (XAI)**: Native integration of `Grad-CAM` backpropagation hooks to generate high-resolution attention heatmaps and overlays on raw imagery.
*   **Stunning Streamlit Dashboard**: A production-grade web UI with deep-blue gradient styling, dynamic KPI metrics, probability bar charts, and interactive toggle sliders.
*   **Clean and Modular Architecture**: Strict separation of concerns (configuration, dataset loader, model generation, training loop, validation/evaluation, inference pipeline, and UI).

---

## 🛠️ Technologies Used

*   **Deep Learning & Math**: `PyTorch`, `Torchvision`, `NumPy`, `SciKit-Learn`
*   **Image Processing**: `Albumentations`, `OpenCV (cv2)`, `Pillow (PIL)`
*   **Dashboard & UI**: `Streamlit`, `Pandas`, `Matplotlib`
*   **Logging**: `TensorBoard` *(Optional)*

---

## 📁 Project Structure

```
C:\Users\arpit\Downloads\IMD Code\
├── app/
│   ├── app.py                      # Main entry point for the Streamlit dashboard (includes custom UI, CSS styling, and Grad-CAM)
│   └── assets/                     # UI assets including weather illustrations and background hero image
├── data/
│   ├── raw/                        # Original, unaugmented dataset categorized by weather classes
│   └── processed/                  # Split directories (train/val/test) and Exploratory Data Analysis (EDA) report outputs
├── models/
│   ├── best_model.pth              # Saved weights of the best performing model (Epoch 8, val accuracy: 96.89%)
│   └── last_checkpoint.pth         # Saved weights of the last epoch (Epoch 14, before early stopping)
├── outputs/
│   └── training/                   # Evaluation plots and metrics (confusion matrix, loss/accuracy curves, metrics.json/csv)
├── src/
│   ├── __init__.py                 # Package initialization
│   ├── config.py                   # Central configurations (hyperparameters, class names, directory paths, normalization stats)
│   ├── dataset.py                  # PyTorch custom dataset class, Albumentations transforms, and DataLoader generators
│   ├── model.py                    # Factory for building EfficientNet-B0 and training parameter count utilities
│   ├── train.py                    # Main model training and validation loop with Early Stopping and learning rate scheduling
│   ├── evaluate.py                 # Core evaluation functions for checkpoints, calculating metrics, and generating reports
│   ├── inference.py                # Command-line/API inference pipeline for running single-image predictions
│   └── utils.py                    # Utility functions (metrics computation, plotting, early stopping, checkpoint saving, JSON/CSV exports)
├── requirements.txt                # Lists external Python packages required for the project
└── README.md                       # Documentation of the project (this file)
```

---

## ⚙️ Installation & Requirements

- **Python Version**: Not specified
- **Operating System Support**: Windows / macOS / Linux (Windows-specific multi-processing protections are handled in config)

To set up the project locally, configure a virtual environment and install the required dependencies:

```powershell
# Step 1: Create a virtual environment
python -m venv venv

# Step 2: Activate the virtual environment
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

# Step 3: Install dependencies
pip install -r requirements.txt
```

---

## 💻 How to Run

Follow this step-by-step execution order to run the pipeline components:

### 1. Data Analysis and Split Verification
Run the Exploratory Data Analysis (EDA) script to generate dataset statistics, verify splits, and export EDA visual distribution reports:
```powershell
python src/analyze_and_split.py
```

### 2. Model Training
Train the classifier on the split dataset using the default configurations. This will use Early Stopping and automatically export the best/last checkpoints to `models/` along with training curves and metrics under `outputs/training/`:
```powershell
python src/train.py
```

### 3. Single-Image CLI Inference
Execute a single-image prediction on any image. The output returns a structured JSON payload with predictions and class probabilities:
```powershell
python src/inference.py --image "data/processed/train/cloudy/cloud10.jpg"
```

### 4. Run the Streamlit Dashboard
Launch the beautiful, interactive dashboard in your web browser:
```powershell
streamlit run app/app.py
```

---

## 📊 Dataset Information

*   **Total Images**: 1,305 valid images
*   **Target Classes (5)**: `cloudy`, `cyclone`, `rainy`, `shine`, `sunrise`

### Dataset Distribution Table

| Class | Raw Total | Train Split | Validation Split | Test Split | Split Validated |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **cloudy** | 336 | 235 | 50 | 51 | Yes |
| **cyclone** | 144 | 100 | 21 | 23 | Yes |
| **rainy** | 215 | 150 | 32 | 33 | Yes |
| **shine** | 253 | 177 | 37 | 39 | Yes |
| **sunrise** | 357 | 249 | 53 | 55 | Yes |
| **TOTAL** | **1,305** | **909** | **193** | **201** | **Yes** |

---

## 🧠 Model Information

*   **Model Backbone**: `EfficientNet-B0` (Pretrained on ImageNet)
*   **Customized Classifier Head**: Dropout layer ($p = 0.2$) -> Linear Layer mapping to 5 classes.
*   **Input Image Dimensions**: $224 \times 224$ pixels
*   **Dynamic Data Normalization Stats**:
    *   Mean: `[0.4611, 0.4577, 0.4499]`
    *   Standard Deviation: `[0.2748, 0.2562, 0.2911]`
*   **Hyperparameters**:
    *   Batch Size: `32`
    *   Initial Learning Rate: `1e-4`
    *   Optimizer: `AdamW` (Weight Decay: `1e-5`)
    *   Learning Rate Scheduler: `ReduceLROnPlateau`
    *   Early Stopping Patience: `5` epochs (early stopping triggered at Epoch 14, best state saved from Epoch 8)
*   **Model Parameters**: Not specified

### Performance Metrics on Test Set

| Metric | Value |
| :--- | :---: |
| **Test Loss** | `0.03236` |
| **Final Accuracy** | `99.00%` |
| **Precision** | `99.05%` |
| **Recall** | `99.00%` |
| **F1-Score** | `99.01%` |

### Confusion Matrix Results
(Rows are true labels, columns are predictions: `cloudy`, `cyclone`, `rainy`, `shine`, `sunrise`)

| Class | cloudy | cyclone | rainy | shine | sunrise |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **cloudy** | **50** | 0 | 0 | 1 | 0 |
| **cyclone** | 0 | **23** | 0 | 0 | 0 |
| **rainy** | 0 | 0 | **33** | 0 | 0 |
| **shine** | 0 | 0 | 0 | **39** | 0 |
| **sunrise** | 0 | 0 | 0 | 1 | **54** |

---

## 🖼️ Screenshots (Placeholder)

### Interactive Dashboard UI

The Streamlit dashboard offers an intuitive user-friendly environment:

| Dashboard Main Interface | Grad-CAM Explainability Overlay |
| :---: | :---: |
| ![Dashboard Home Page](app/assets/weather.png) | *[Grad-CAM Attention Overlay Placeholder]* |

---

## 🔮 Future Improvements

1.  **CUDA/GPU Integration**: Enable automatic switching to GPU devices for faster training (current training took ~22 minutes on CPU).
2.  **Model Diversification**: Support additional model backbones (e.g., ResNet-50, ConvNeXt, Vision Transformers) via `src/model.py`.
3.  **Expanded Categorization**: Collect and pre-process additional weather imagery to include labels for Fog, Hail, Sandstorms, and Blizzard conditions.
4.  **Live Satellite Stream Integration**: Integrate real-time API imagery fetches from meteorological satellites to provide real-time cloud analysis.

---

## 👥 Contributors

- **Arpit** – AI Model Development, Model Training, Data Preprocessing, Backend Integration, Project Integration
- **Harman Singh** – UI/UX Design and Streamlit User Interface Development

---

## 📄 License

Not specified
