# Phase 3 Report: Streamlit Deployment & Explainable AI (Grad-CAM)

Generated: 2026-07-18

This phase documents the deployment of the 15-class weather classification system via a modernized Streamlit application, incorporating explainable AI, batch processing, and a dynamic model comparison dashboard.

---

## 🚀 Streamlit Deployment Details

* **Deployment URL**: Running locally at `http://localhost:8501`.
* **Pipeline Caching**: Implemented `@st.cache_resource` for the `WeatherInferencePipeline` to load and cache models into GPU memory dynamically, enabling instantaneous model switching and dual predictions.
* **Checkpoint Fallback**: Dynamically routes checkpoint loading:
  * For EfficientNet-B0: Checks for `models/efficientnet_b0_15class_best.pth`, falling back to `models/best_model.pth` if unavailable.
  * For Swin Transformer: Checks for `models/swin_transformer_15class_best.pth`.

---

## 🎨 UI Modernization Features

* **Sidebar Navigation**: Added a sidebar radio selection to switch pages:
  * `Single Image Analysis` (Primary dashboard interface).
  * `Batch Classification` (Bulk image classifier).
  * `Model Comparison Dashboard` (Comparative analytics).
* **Responsive Layout**: Designed a two-column responsive layout utilizing streamlined metric cards, prediction hero banners, top-3 class confidence outputs, and confidence bar charts.
* **Session Prediction History**: Tracks all single-image predictions in the current session inside `st.session_state` and renders them in a table.

---

## 📁 Batch Classification Workflow

* **Automatic Directory Creation**: Automatically generates folder structures for all 15 classes under `classified_images/` when the batch process starts.
* **Overwriting Protection**: Generates unique names for saved images using a UUID suffix (e.g. `image_a1b2c3.jpg` instead of overwriting existing files in the class folders).
* **Local Copy Preservation**: Reads raw files in memory and writes copies directly to local directories, preserving the original uploaded image file paths.
* **Performance Summary**: Displays a progress bar during processing, followed by summary metric cards showing:
  * Total uploaded images
  * Successfully classified images
  * Failed images count
  * Processing time in seconds
  * Class-wise distribution count

---

## 🧠 Explainable AI (Grad-CAM)

* **Architecture**: Configured Grad-CAM targeting the final feature extraction layer of **EfficientNet-B0** (`features[-1]`).
* **Visualizations**: Displays the raw heatmap and overlays it on the original RGB image (using a 60% image / 40% heatmap blend) side-by-side.
* **Availability**: Controlled via a checkbox, hidden/disabled automatically when the Swin Transformer model is selected.

---

## 🎛️ Model Selector & Dual-Model Comparison

* **Dynamic Model Selector**: Select between **EfficientNet-B0 (Default)** and **Swin Transformer** in the sidebar.
* **Optional Dual Prediction**: Enabled via checkbox. Runs inference on both models, measures GPU latency per model in milliseconds, and outputs a comparative table side-by-side.

---

## 📊 Model Comparison Dashboard Details

* **Dynamic Metric Loading**: Reads evaluation stats directly from the saved JSON files (`metrics.json` and `swin/metrics.json`), presenting the table dynamically.
* **Training History**: Renders training curves (loss and accuracy) side-by-side for both architectures.
* **Architecture Overview**: Detailed cards outlining pros and cons.
* **Deployment Recommendation**: Explains the deployment selection of **EfficientNet-B0** due to high efficiency (6.8x smaller size, 43.5% faster inference, 3.5x faster training).

---

## 📝 Verification Checklist

* [x] **Local Server Binding**: Starts and binds to `http://localhost:8501`.
* [x] **Model Loading**: Correctly caches and switches model states in GPU memory.
* [x] **Grad-CAM**: Renders heatmap and overlay accurately on EfficientNet-B0.
* [x] **Dual Prediction**: Successfully compares models and calculates GPU latencies.
* [x] **Batch Classifier**: Creates folders, copies unique files, and displays processing metrics.
* [x] **Dashboard Parsing**: Dynamically loads validation metrics.

---

## ⚠️ Remaining Known Limitations

* **Swin Grad-CAM Support**: Grad-CAM target layers are architecturally incompatible with Swin Transformer's attention structures. The app restricts Grad-CAM features to the EfficientNet-B0 backbone.
* **Session Persistence**: Prediction history list is cached in Streamlit's `st.session_state` and will reset when the browser tab is refreshed.
