import csv
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_enabled_for_device(device):
    return device.type == "cuda"


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path, data):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def save_csv(path, rows, fieldnames):
    ensure_dir(Path(path).parent)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_summary_writer(log_dir):
    if SummaryWriter is None:
        print("TensorBoard is unavailable because the tensorboard package is not installed.")
        return None
    ensure_dir(log_dir)
    return SummaryWriter(log_dir=log_dir)


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.should_stop = False

    def step(self, validation_loss):
        if self.best_loss is None or validation_loss < self.best_loss - self.min_delta:
            self.best_loss = validation_loss
            self.counter = 0
            return False

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return self.should_stop


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_loss, classes, metrics=None):
    ensure_dir(Path(path).parent)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_val_loss": best_val_loss,
        "classes": classes,
        "metrics": metrics or {},
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location=None):
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def calculate_classification_metrics(y_true, y_pred, class_names):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        ),
    }


def save_metrics_outputs(metrics, output_dir):
    ensure_dir(output_dir)
    save_json(Path(output_dir) / "metrics.json", metrics)

    summary_rows = [
        {"metric": "accuracy", "value": metrics["accuracy"]},
        {"metric": "precision", "value": metrics["precision"]},
        {"metric": "recall", "value": metrics["recall"]},
        {"metric": "f1_score", "value": metrics["f1_score"]},
    ]
    save_csv(Path(output_dir) / "metrics.csv", summary_rows, ["metric", "value"])


def plot_training_curves(history, output_dir):
    ensure_dir(output_dir)
    epochs = range(1, len(history.get("train_loss", [])) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.get("train_loss", []), label="Train Loss")
    plt.plot(epochs, history.get("val_loss", []), label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "loss_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.get("train_accuracy", []), label="Train Accuracy")
    plt.plot(epochs, history.get("val_accuracy", []), label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "accuracy_curve.png", dpi=150)
    plt.close()


def plot_confusion_matrix(cm, class_names, output_path):
    ensure_dir(Path(output_path).parent)
    matrix = np.array(cm)
    plt.figure(figsize=(8, 7))
    plt.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    threshold = matrix.max() / 2.0 if matrix.size and matrix.max() > 0 else 0
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            plt.text(
                col_index,
                row_index,
                int(matrix[row_index, col_index]),
                ha="center",
                va="center",
                color="white" if matrix[row_index, col_index] > threshold else "black",
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def history_to_rows(history):
    rows = []
    for index in range(len(history.get("train_loss", []))):
        rows.append(
            {
                "epoch": index + 1,
                "train_loss": history["train_loss"][index],
                "val_loss": history["val_loss"][index],
                "train_accuracy": history["train_accuracy"][index],
                "val_accuracy": history["val_accuracy"][index],
            }
        )
    return rows
