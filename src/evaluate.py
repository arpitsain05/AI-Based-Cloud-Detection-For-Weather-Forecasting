from pathlib import Path

import torch

try:
    from src.config import CLASSES, IMAGE_SIZE, MODEL_NAME, NUM_CLASSES
    from src.model import build_model
    from src.utils import calculate_classification_metrics, plot_confusion_matrix, save_metrics_outputs
except ImportError:
    from config import CLASSES, IMAGE_SIZE, MODEL_NAME, NUM_CLASSES
    from model import build_model
    from utils import calculate_classification_metrics, plot_confusion_matrix, save_metrics_outputs


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            predictions = torch.argmax(outputs, dim=1)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            correct += (predictions == labels).sum().item()
            total += batch_size
            all_predictions.extend(predictions.cpu().tolist())
            all_targets.extend(labels.cpu().tolist())

    average_loss = running_loss / total if total else 0.0
    accuracy = correct / total if total else 0.0
    return average_loss, accuracy, all_targets, all_predictions


def evaluate_checkpoint(checkpoint_path, data_loader, criterion, device, output_dir, class_names=CLASSES):
    model = build_model(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        pretrained=False,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    loss, accuracy, y_true, y_pred = evaluate_model(model, data_loader, criterion, device)
    metrics = calculate_classification_metrics(y_true, y_pred, class_names)
    metrics["loss"] = loss
    metrics["accuracy"] = accuracy

    output_dir = Path(output_dir)
    save_metrics_outputs(metrics, output_dir)
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, output_dir / "confusion_matrix.png")
    return metrics
