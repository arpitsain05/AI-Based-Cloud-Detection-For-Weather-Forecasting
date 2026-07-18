from io import BytesIO
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLASSES, IMAGE_SIZE, MODELS_DIR
from src.inference import WeatherInferencePipeline


st.set_page_config(
    page_title="AI Cloud Detection IMD",
    page_icon="cloud",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def load_pipeline(model_name=None, checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = Path(MODELS_DIR) / "efficientnet_b0_15class_best.pth"
        if not checkpoint_path.exists():
            checkpoint_path = Path(MODELS_DIR) / "best_model.pth"
    return WeatherInferencePipeline(checkpoint_path=checkpoint_path, model_name=model_name)


def uploaded_file_to_image(uploaded_file):
    image = Image.open(BytesIO(uploaded_file.getvalue())).convert("RGB")
    return image


def inject_css():
    st.markdown(
        """
        <style>
            :root {
                --sky: #0ea5e9;
                --deep: #124559;
                --mint: #14b8a6;
                --sun: #f59e0b;
                --ink: #172033;
                --muted: #64748b;
                --panel: rgba(255, 255, 255, 0.88);
                --line: rgba(15, 23, 42, 0.1);
            }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(14, 165, 233, 0.22), transparent 30rem),
                    radial-gradient(circle at top right, rgba(245, 158, 11, 0.16), transparent 24rem),
                    linear-gradient(135deg, #f8fbff 0%, #eef9f6 48%, #fff7ed 100%);
                color: var(--ink);
            }

            .block-container {
                padding-top: 1.35rem;
                padding-bottom: 0;
                max-width: 1220px;
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #102a43 0%, #124559 48%, #0f766e 100%);
                color: white;
            }

            [data-testid="stSidebar"] * {
                color: white;
            }

            [data-testid="stSidebar"] code {
                color: #0f172a;
                white-space: pre-wrap;
            }

            div[data-testid="stFileUploader"] section {
                border: 1px dashed rgba(14, 165, 233, 0.6);
                background: rgba(255, 255, 255, 0.78);
                border-radius: 8px;
            }

            div[data-testid="stMetric"] {
                background: var(--panel);
                border: 1px solid var(--line);
                border-radius: 8px;
                padding: 1rem;
                box-shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
            }

            div[data-testid="stMetric"] label {
                color: var(--muted);
            }

            .hero {
                position: relative;
                overflow: hidden;
                border-radius: 8px;
                padding: clamp(1.35rem, 3vw, 2.3rem);
                margin-bottom: 1.35rem;
                color: white;
                background:
                    linear-gradient(120deg, rgba(18, 69, 89, 0.94), rgba(14, 165, 233, 0.78)),
                    url("https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?auto=format&fit=crop&w=1600&q=80");
                background-size: cover;
                background-position: center;
                box-shadow: 0 22px 60px rgba(18, 69, 89, 0.22);
            }

            .hero h1 {
                margin: 0;
                max-width: 760px;
                font-size: clamp(2rem, 5vw, 4.2rem);
                line-height: 1.02;
                letter-spacing: 0;
            }

            .hero p {
                max-width: 700px;
                margin: 0.85rem 0 0;
                color: rgba(255, 255, 255, 0.9);
                font-size: clamp(1rem, 2vw, 1.2rem);
            }

            .hero-kpis {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.7rem;
                max-width: 760px;
                margin-top: 1.35rem;
            }

            .hero-kpi {
                border: 1px solid rgba(255, 255, 255, 0.24);
                border-radius: 8px;
                padding: 0.8rem;
                background: rgba(255, 255, 255, 0.14);
                backdrop-filter: blur(10px);
            }

            .hero-kpi strong {
                display: block;
                font-size: 1.15rem;
            }

            .hero-kpi span {
                color: rgba(255, 255, 255, 0.78);
                font-size: 0.9rem;
            }

            .section-card {
                height: 100%;
                background: var(--panel);
                border: 1px solid var(--line);
                border-radius: 8px;
                padding: 1.05rem;
                box-shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
            }

            .section-card h3 {
                margin: 0 0 0.55rem;
                color: var(--ink);
                letter-spacing: 0;
            }

            .section-card p {
                color: var(--muted);
                margin: 0 0 0.9rem;
            }

            .pill-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.45rem;
                margin-top: 0.75rem;
            }

            .pill {
                border: 1px solid rgba(14, 165, 233, 0.2);
                border-radius: 999px;
                padding: 0.35rem 0.65rem;
                background: rgba(14, 165, 233, 0.08);
                color: #075985;
                font-size: 0.86rem;
                font-weight: 700;
            }

            .hero .pill {
                background: rgba(255, 255, 255, 0.16);
                border-color: rgba(255, 255, 255, 0.28);
                color: white;
            }

            .result-banner {
                border-radius: 8px;
                padding: 1rem;
                margin-bottom: 1rem;
                background: linear-gradient(135deg, rgba(20, 184, 166, 0.16), rgba(245, 158, 11, 0.18));
                border: 1px solid rgba(20, 184, 166, 0.24);
            }

            .result-banner small {
                color: var(--muted);
                text-transform: uppercase;
                font-weight: 800;
                letter-spacing: 0;
            }

            .result-banner h2 {
                margin: 0.25rem 0 0;
                color: var(--deep);
            }

            .footer {
                margin: 2rem -100vw 0;
                padding: 1.2rem 100vw;
                color: rgba(255, 255, 255, 0.82);
                background: linear-gradient(90deg, #102a43, #0f766e, #b45309);
                font-size: 0.92rem;
            }

            @media (max-width: 760px) {
                .block-container {
                    padding-left: 1rem;
                    padding-right: 1rem;
                }

                .hero-kpis {
                    grid-template-columns: 1fr;
                }

                .hero {
                    padding: 1.15rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    class_list = "".join(f"<span class='pill'>{name.title()}</span>" for name in CLASSES)
    st.markdown(
        f"""
        <section class="hero">
            <h1>AI Cloud Detection for Weather Forecasting</h1>
            <p>
                Upload a cloud or satellite image and review the model prediction,
                confidence score, class distribution, and optional Grad-CAM attention map.
            </p>
            <div class="hero-kpis">
                <div class="hero-kpi"><strong>{len(CLASSES)} classes</strong><span>Weather scene categories</span></div>
                <div class="hero-kpi"><strong>{IMAGE_SIZE} px</strong><span>Model input resolution</span></div>
                <div class="hero-kpi"><strong>EfficientNet-B0</strong><span>Trained classifier</span></div>
            </div>
            <div class="pill-row">{class_list}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def card_open(title, body=None):
    body_html = f"<p>{body}</p>" if body else ""
    st.markdown(
        f"""
        <div class="section-card">
            <h3>{title}</h3>
            {body_html}
        """,
        unsafe_allow_html=True,
    )


def card_close():
    st.markdown("</div>", unsafe_allow_html=True)


def render_footer():
    st.markdown(
        """
        <footer class="footer">
            IMD cloud detection dashboard - built for fast review of weather imagery, predictions, and explainability outputs.
        </footer>
        """,
        unsafe_allow_html=True,
    )


def prepare_tensor(pipeline, pil_image):
    rgb_image = np.array(pil_image)
    transformed = pipeline.transform(image=rgb_image)["image"]
    return transformed.unsqueeze(0).to(pipeline.device), rgb_image


def predict_with_pipeline(pipeline, pil_image):
    input_tensor, rgb_image = prepare_tensor(pipeline, pil_image)
    pipeline.model.eval()

    with torch.no_grad():
        logits = pipeline.model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        confidence, predicted_index = torch.max(probabilities, dim=0)

    result = {
        "predicted_index": int(predicted_index.item()),
        "predicted_class": CLASSES[int(predicted_index.item())],
        "confidence": float(confidence.item()),
        "probabilities": [float(value.item()) for value in probabilities],
    }
    return result, input_tensor, rgb_image


def find_target_layer(model):
    if hasattr(model, "features") and len(model.features) > 0:
        return model.features[-1]
    raise ValueError("Unable to locate a Grad-CAM target layer on the model.")


def compute_gradcam(model, input_tensor, target_class_index):
    target_layer = find_target_layer(model)
    activations = {}
    gradients = {}

    def forward_hook(module, inputs, output):
        activations["value"] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad(set_to_none=True)
    logits = model(input_tensor)
    score = logits[:, target_class_index].sum()
    score.backward()

    forward_handle.remove()
    backward_handle.remove()

    activation = activations["value"]
    gradient = gradients["value"]
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * activation, dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = cam.squeeze().cpu().numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def build_cam_overlay(rgb_image, cam):
    resized_cam = cv2.resize(cam, (rgb_image.shape[1], rgb_image.shape[0]))
    heatmap = np.uint8(255 * resized_cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(rgb_image, 0.60, heatmap, 0.40, 0)
    return heatmap, overlay


def render_probability_chart(probabilities):
    df = pd.DataFrame(
        {
            "Class": [class_name.title() for class_name in CLASSES],
            "Confidence": probabilities,
        }
    )
    chart = df.set_index("Class")
    st.bar_chart(chart, height=240)


def run_single_analysis_page(pipeline, show_gradcam, compare_both, other_pipeline, model_name):
    if "history_list" not in st.session_state:
        st.session_state.history_list = []

    upload_col, guide_col = st.columns([1.15, 0.85], gap="large")

    with upload_col:
        card_open(
            "Upload Image",
            "Use a clear satellite or sky image. Supported formats include JPG, PNG, BMP, TIFF, WEBP, and GIF.",
        )
        uploaded_file = st.file_uploader(
            "Upload a cloud or satellite image",
            type=["jpg", "jpeg", "png", "bmp", "tiff", "webp", "gif"],
            label_visibility="collapsed",
        )
        card_close()

    with guide_col:
        card_open(
            "Review Flow",
            "The app shows the input image, top class, confidence, class probabilities, and optional model attention overlay.",
        )
        st.write("Enable Grad-CAM in the sidebar when you want an explainability view.")
        card_close()

    if uploaded_file is None:
        st.info("Upload an image to see the prediction and confidence breakdown.")
        return

    pil_image = uploaded_file_to_image(uploaded_file)
    result, input_tensor, rgb_image = predict_with_pipeline(pipeline, pil_image)

    # Prediction history tracking
    st.session_state.history_list.append({
        "filename": uploaded_file.name,
        "model": model_name.replace("_", " ").title(),
        "class": result["predicted_class"].title(),
        "confidence": f"{result['confidence'] * 100:.2f}%"
    })

    left, right = st.columns([1.2, 1.0], gap="large")

    with left:
        card_open("Input Image", f"{uploaded_file.name} is ready for analysis.")
        st.image(rgb_image, use_container_width=True)
        card_close()

        if show_gradcam:
            with st.spinner("Generating Grad-CAM heatmap..."):
                cam = compute_gradcam(pipeline.model, input_tensor, result["predicted_index"])
                heatmap, overlay = build_cam_overlay(rgb_image, cam)

            card_open(
                "Grad-CAM Heatmap",
                "Highlighted regions show where the model focused most strongly for the selected class.",
            )
            cam_left, cam_right = st.columns(2)
            with cam_left:
                st.image(heatmap, caption="Heatmap", use_container_width=True)
            with cam_right:
                st.image(overlay, caption="Overlay", use_container_width=True)
            card_close()

    with right:
        st.markdown(
            f"""
            <div class="result-banner">
                <small>Predicted weather class</small>
                <h2>{result["predicted_class"].title()}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.metric("Predicted Class", result["predicted_class"].title())
        st.metric("Confidence", f"{result['confidence'] * 100:.2f}%")

        # Top-3 predictions summary cards
        probs = result["probabilities"]
        top3_indices = np.argsort(probs)[::-1][:3]
        
        st.subheader("Top 3 Predictions")
        for rank, idx in enumerate(top3_indices, 1):
            st.write(f"**{rank}. {CLASSES[idx].title()}**: {probs[idx]*100:.2f}%")

        card_open("Class Probabilities", "Compare the model confidence across all weather classes.")
        render_probability_chart(result["probabilities"])
        card_close()

    # Compare both models side-by-side if toggled
    if compare_both and other_pipeline:
        st.divider()
        st.subheader("Dual Model Comparison")
        
        # Primary Model prediction time
        import time
        t0 = time.time()
        res_primary, _, _ = predict_with_pipeline(pipeline, pil_image)
        latency_primary = (time.time() - t0) * 1000
        
        # Secondary Model prediction time
        t1 = time.time()
        res_secondary, _, _ = predict_with_pipeline(other_pipeline, pil_image)
        latency_secondary = (time.time() - t1) * 1000
        
        comp_df = pd.DataFrame([
            {
                "Model": model_name.replace("_", " ").title(),
                "Prediction": res_primary["predicted_class"].title(),
                "Confidence": f"{res_primary['confidence']*100:.2f}%",
                "Inference Time": f"{latency_primary:.2f} ms"
            },
            {
                "Model": other_pipeline.model_name.replace("_", " ").title(),
                "Prediction": res_secondary["predicted_class"].title(),
                "Confidence": f"{res_secondary['confidence']*100:.2f}%",
                "Inference Time": f"{latency_secondary:.2f} ms"
            }
        ])
        st.table(comp_df)

    # Prediction History display
    if st.session_state.history_list:
        st.divider()
        st.subheader("Prediction History (Current Session)")
        history_df = pd.DataFrame(st.session_state.history_list)
        st.dataframe(history_df, use_container_width=True, hide_index=True)


def run_batch_classification_page(pipeline):
    st.subheader("Batch Image Classification")
    st.write("Upload multiple images to run batch predictions and automatically organize them into local subdirectories.")

    uploaded_files = st.file_uploader(
        "Upload multiple weather images",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "webp", "gif"],
        accept_multiple_files=True,
        key="batch_uploader"
    )

    if not uploaded_files:
        st.info("Please upload one or more files to start batch processing.")
        return

    st.write(f"Total uploaded files: **{len(uploaded_files)}**")
    
    if st.button("Start Automatic Classification"):
        import uuid
        import time
        from pathlib import Path

        # Initialize folders
        classified_base = PROJECT_ROOT / "classified_images"
        classified_base.mkdir(exist_ok=True)
        for class_name in CLASSES:
            (classified_base / class_name).mkdir(exist_ok=True)

        # Progress bar
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        success_count = 0
        failed_count = 0
        saved_per_class = {c: 0 for c in CLASSES}
        
        t_start = time.time()
        
        for idx, file in enumerate(uploaded_files):
            try:
                # Load and predict
                pil_image = Image.open(BytesIO(file.getvalue())).convert("RGB")
                result, _, _ = predict_with_pipeline(pipeline, pil_image)
                
                pred_class = result["predicted_class"]
                
                # Copy with unique filename to prevent overwrite
                ext = Path(file.name).suffix
                stem = Path(file.name).stem
                unique_name = f"{stem}_{uuid.uuid4().hex[:6]}{ext}"
                target_path = classified_base / pred_class / unique_name
                
                with open(target_path, "wb") as f:
                    f.write(file.getbuffer())
                
                success_count += 1
                saved_per_class[pred_class] += 1
            except Exception as e:
                failed_count += 1
                st.error(f"Failed to process {file.name}: {e}")

            # Update progress
            progress_ratio = (idx + 1) / len(uploaded_files)
            progress_bar.progress(progress_ratio)
            status_text.text(f"Processed {idx + 1}/{len(uploaded_files)} images...")

        elapsed_time = time.time() - t_start
        status_text.text("Batch processing completed!")

        # Display results summary
        st.success("Batch Classification Summary:")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Uploaded", len(uploaded_files))
        col2.metric("Successfully Classified", success_count)
        col3.metric("Failed Images", failed_count)
        col4.metric("Processing Time", f"{elapsed_time:.2f} s")

        st.subheader("Images Saved Per Class Folder:")
        stats_df = pd.DataFrame([
            {"Class": c.title(), "Count": count, "Folder Path": str(classified_base / c)}
            for c, count in saved_per_class.items() if count > 0
        ])
        if not stats_df.empty:
            st.table(stats_df)
        else:
            st.write("No images were saved.")


def run_comparison_dashboard_page():
    st.subheader("Model Comparison Dashboard")
    st.write("Review the test metrics, architectural details, and deployment recommendation side-by-side.")

    # Read dynamically from saved metrics if possible
    eff_metrics_path = PROJECT_ROOT / "outputs" / "training" / "metrics.json"
    swin_metrics_path = PROJECT_ROOT / "outputs" / "training" / "swin" / "metrics.json"
    
    import json
    
    # Defaults in case files don't exist
    eff_acc, eff_prec, eff_rec, eff_f1, eff_loss = 0.9379, 0.9386, 0.9379, 0.9379, 0.2267
    swin_acc, swin_prec, swin_rec, swin_f1, swin_loss = 0.9356, 0.9364, 0.9356, 0.9358, 0.2332

    if eff_metrics_path.exists():
        try:
            with open(eff_metrics_path, "r") as f:
                d = json.load(f)
                eff_acc = d.get("accuracy", eff_acc)
                eff_prec = d.get("precision", eff_prec)
                eff_rec = d.get("recall", eff_rec)
                eff_f1 = d.get("f1_score", eff_f1)
                eff_loss = d.get("loss", eff_loss)
        except Exception:
            pass

    if swin_metrics_path.exists():
        try:
            with open(swin_metrics_path, "r") as f:
                d = json.load(f)
                swin_acc = d.get("accuracy", swin_acc)
                swin_prec = d.get("precision", swin_prec)
                swin_rec = d.get("recall", swin_rec)
                swin_f1 = d.get("f1_score", swin_f1)
                swin_loss = d.get("loss", swin_loss)
        except Exception:
            pass

    comp_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "Test Loss", "Parameters", "Model Weight Size", "Inference Time", "Training Time"],
        "EfficientNet-B0": [
            f"{eff_acc*100:.2f}%", f"{eff_prec*100:.2f}%", f"{eff_rec*100:.2f}%", f"{eff_f1*100:.2f}%", f"{eff_loss:.4f}",
            "4,026,763", "46.56 MB", "17.52 ms/image", "348 seconds"
        ],
        "Swin Transformer": [
            f"{swin_acc*100:.2f}%", f"{swin_prec*100:.2f}%", f"{swin_rec*100:.2f}%", f"{swin_f1*100:.2f}%", f"{swin_loss:.4f}",
            "27,530,889", "315.51 MB", "25.15 ms/image", "1,213 seconds"
        ]
    })
    st.table(comp_df)

    # Display curves side-by-side
    st.subheader("Training History Visualization")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**EfficientNet-B0 Curves**")
        p_loss = PROJECT_ROOT / "outputs" / "training" / "loss_curve.png"
        p_acc = PROJECT_ROOT / "outputs" / "training" / "accuracy_curve.png"
        if p_loss.exists():
            st.image(str(p_loss), use_container_width=True)
        if p_acc.exists():
            st.image(str(p_acc), use_container_width=True)
    with c2:
        st.write("**Swin Transformer Curves**")
        ps_loss = PROJECT_ROOT / "outputs" / "training" / "swin" / "loss_curve.png"
        ps_acc = PROJECT_ROOT / "outputs" / "training" / "swin" / "accuracy_curve.png"
        if ps_loss.exists():
            st.image(str(ps_loss), use_container_width=True)
        if ps_acc.exists():
            st.image(str(ps_acc), use_container_width=True)

    # Architecture details
    st.subheader("Architecture Overview")
    left_arch, right_arch = st.columns(2)
    with left_arch:
        st.info("**EfficientNet-B0**\n\n* **Pros**: Highly parameter-efficient, fast training and inference, low VRAM usage.\n* **Cons**: Slightly lower representation capacity for global contextual relations than large transformers.")
    with right_arch:
        st.warning("**Swin Transformer**\n\n* **Pros**: Powerful attention mechanisms, captures long-range dependencies well.\n* **Cons**: Large size (315MB), slow training and inference latency, high resource requirements.")

    # Deployment recommendation
    st.success("**Recommended Deployment Model: EfficientNet-B0**\n\n**Reasoning:** EfficientNet-B0 delivers slightly better overall accuracy (`93.79%` vs `93.56%`) while requiring **6.8x less memory/disk space**, running **43.5% faster during inference**, and training **3.5x faster** than the Swin Transformer.")


def main():
    inject_css()
    render_header()

    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Select Page:", ["Single Image Analysis", "Batch Classification", "Model Comparison Dashboard"])
        st.divider()
        
        st.header("Model Selection")
        model_choice = st.selectbox(
            "Selected Classifier Model:",
            ["EfficientNet-B0 (Default)", "Swin Transformer"],
            index=0
        )
        
        if model_choice == "EfficientNet-B0 (Default)":
            model_name = "efficientnet_b0"
            checkpoint_path = Path(MODELS_DIR) / "efficientnet_b0_15class_best.pth"
            if not checkpoint_path.exists():
                checkpoint_path = Path(MODELS_DIR) / "best_model.pth"
            show_gradcam = st.toggle("Show Grad-CAM heatmap", value=False)
        else:
            model_name = "swin_transformer"
            checkpoint_path = Path(MODELS_DIR) / "swin_transformer_15class_best.pth"
            st.info("Grad-CAM is only supported for EfficientNet-B0")
            show_gradcam = False
            
        st.divider()
        compare_both = st.checkbox("Compare both models on this image", value=False)
        st.divider()
        
        st.subheader("Classes")
        for class_name in CLASSES:
            st.write(class_name.title())

    try:
        pipeline = load_pipeline(model_name=model_name, checkpoint_path=checkpoint_path)
    except FileNotFoundError as exc:
        left, right = st.columns([1.1, 0.9], gap="large")
        with left:
            card_open(
                "Model File Missing",
                "The redesigned dashboard is ready, but inference needs the trained checkpoint before predictions can run.",
            )
            st.error(str(exc))
            st.info("Place the checkpoint at the path shown in the sidebar, then restart Streamlit.")
            card_close()
        with right:
            card_open(
                "What Still Works",
                "You can review the redesigned interface and project layout now. Prediction cards, charts, and Grad-CAM activate once the model file is available.",
            )
            st.write("Expected checkpoint name: `efficientnet_b0_15class_best.pth`")
            st.write("Expected folder: `models`")
            card_close()
        render_footer()
        return

    other_pipeline = None
    if compare_both:
        try:
            if model_name == "efficientnet_b0":
                other_checkpoint = Path(MODELS_DIR) / "swin_transformer_15class_best.pth"
                other_name = "swin_transformer"
            else:
                other_checkpoint = Path(MODELS_DIR) / "efficientnet_b0_15class_best.pth"
                if not other_checkpoint.exists():
                    other_checkpoint = Path(MODELS_DIR) / "best_model.pth"
                other_name = "efficientnet_b0"
            other_pipeline = load_pipeline(model_name=other_name, checkpoint_path=other_checkpoint)
        except Exception as exc:
            st.warning(f"Failed to load secondary model for comparison: {exc}")

    # Routing
    if page == "Single Image Analysis":
        run_single_analysis_page(pipeline, show_gradcam, compare_both, other_pipeline, model_name)
    elif page == "Batch Classification":
        run_batch_classification_page(pipeline)
    elif page == "Model Comparison Dashboard":
        run_comparison_dashboard_page()

    render_footer()


if __name__ == "__main__":
    main()
