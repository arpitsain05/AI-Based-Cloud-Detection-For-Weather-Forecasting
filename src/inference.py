import argparse
import json
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F

try:
    from src.config import CLASSES, IMAGE_SIZE, MODELS_DIR, MODEL_NAME, NUM_CLASSES, NORM_MEAN, NORM_STD
    from src.dataset import get_transforms
    from src.model import build_model
except ImportError:
    from config import CLASSES, IMAGE_SIZE, MODELS_DIR, MODEL_NAME, NUM_CLASSES, NORM_MEAN, NORM_STD
    from dataset import get_transforms
    from model import build_model


DEFAULT_CHECKPOINT_PATH = Path(MODELS_DIR) / "best_model.pth"


class WeatherInferencePipeline:
    """
    Load the trained weather classifier and run single-image inference.
    Uses the same resize/normalization pipeline as validation/testing.
    """

    def __init__(self, checkpoint_path=DEFAULT_CHECKPOINT_PATH, device=None):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        _, self.transform = get_transforms(img_size=IMAGE_SIZE, mean=NORM_MEAN, std=NORM_STD)

    def _load_model(self):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.checkpoint_path}")

        model = build_model(model_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=False)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def _load_image(self, image_path):
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def predict_image(self, image_path):
        image = self._load_image(image_path)
        transformed = self.transform(image=image)["image"]
        batch = transformed.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(batch)
            probabilities = F.softmax(logits, dim=1)
            confidence, predicted_index = torch.max(probabilities, dim=1)

        predicted_index = predicted_index.item()
        confidence = confidence.item()
        predicted_class = CLASSES[predicted_index]

        return {
            "image_path": str(Path(image_path)),
            "predicted_class": predicted_class,
            "confidence": confidence,
            "predicted_index": predicted_index,
            "class_probabilities": {
                class_name: float(probabilities[0, index].item())
                for index, class_name in enumerate(CLASSES)
            },
        }


def main():
    parser = argparse.ArgumentParser(description="Run inference on a single weather image.")
    parser.add_argument("--image", required=True, help="Path to the image to classify.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT_PATH), help="Path to the trained model checkpoint.")
    args = parser.parse_args()

    pipeline = WeatherInferencePipeline(checkpoint_path=args.checkpoint)
    result = pipeline.predict_image(args.image)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
