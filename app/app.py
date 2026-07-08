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
    page_icon="☁",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def load_pipeline():
    return WeatherInferencePipeline()


def uploaded_file_to_image(uploaded_file):
    image = Image.open(BytesIO(uploaded_file.getvalue())).convert("RGB")
    return image


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
            "Class": CLASSES,
            "Confidence": probabilities,
        }
    )
    chart = df.set_index("Class")
    st.bar_chart(chart, height=240)


def main():
    st.title("AI Cloud Detection IMD")
    st.caption("Upload a satellite/cloud image to classify it with the trained EfficientNet-B0 model.")

    with st.sidebar:
        st.header("Inference Settings")
        show_gradcam = st.toggle("Show Grad-CAM heatmap", value=False)
        st.write("Model checkpoint")
        st.code(str(Path(MODELS_DIR) / "best_model.pth"), language="text")
        st.write("Classes")
        st.write(", ".join(CLASSES))

    pipeline = load_pipeline()

    uploaded_file = st.file_uploader(
        "Upload a cloud or satellite image",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "webp", "gif"],
    )

    if uploaded_file is None:
        st.info("Upload an image to see the prediction and confidence breakdown.")
        return

    pil_image = uploaded_file_to_image(uploaded_file)
    result, input_tensor, rgb_image = predict_with_pipeline(pipeline, pil_image)

    left, right = st.columns([1.2, 1.0], gap="large")

    with left:
        st.subheader("Input Image")
        st.image(rgb_image, use_column_width=True)

        if show_gradcam:
            with st.spinner("Generating Grad-CAM heatmap..."):
                cam = compute_gradcam(pipeline.model, input_tensor, result["predicted_index"])
                heatmap, overlay = build_cam_overlay(rgb_image, cam)

            st.subheader("Grad-CAM Heatmap")
            cam_left, cam_right = st.columns(2)
            with cam_left:
                st.image(heatmap, caption="Heatmap", use_column_width=True)
            with cam_right:
                st.image(overlay, caption="Overlay", use_column_width=True)

    with right:
        st.subheader("Prediction")
        st.metric("Predicted Class", result["predicted_class"])
        st.metric("Confidence", f"{result['confidence'] * 100:.2f}%")

        st.subheader("Class Probabilities")
        render_probability_chart(result["probabilities"])

        probability_table = pd.DataFrame(
            {
                "Class": CLASSES,
                "Probability": [f"{value * 100:.2f}%" for value in result["probabilities"]],
            }
        )
        st.dataframe(probability_table, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
