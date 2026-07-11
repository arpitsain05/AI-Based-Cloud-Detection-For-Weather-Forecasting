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
def load_pipeline():
    return WeatherInferencePipeline()


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


def main():
    inject_css()
    render_header()

    with st.sidebar:
        st.header("Dashboard Controls")
        st.caption("Tune the prediction view before uploading imagery.")
        show_gradcam = st.toggle("Show Grad-CAM heatmap", value=False)
        st.divider()
        st.subheader("Model Checkpoint")
        st.code(str(Path(MODELS_DIR) / "best_model.pth"), language="text")
        st.subheader("Classes")
        for class_name in CLASSES:
            st.write(class_name.title())

    try:
        pipeline = load_pipeline()
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
            st.write("Expected checkpoint name: `best_model.pth`")
            st.write("Expected folder: `models`")
            card_close()
        render_footer()
        return

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
        render_footer()
        return

    pil_image = uploaded_file_to_image(uploaded_file)
    result, input_tensor, rgb_image = predict_with_pipeline(pipeline, pil_image)

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

        card_open("Class Probabilities", "Compare the model confidence across all weather classes.")
        render_probability_chart(result["probabilities"])

        probability_table = pd.DataFrame(
            {
                "Class": [class_name.title() for class_name in CLASSES],
                "Probability": [f"{value * 100:.2f}%" for value in result["probabilities"]],
            }
        )
        st.dataframe(probability_table, use_container_width=True, hide_index=True)
        card_close()

    render_footer()


if __name__ == "__main__":
    main()
