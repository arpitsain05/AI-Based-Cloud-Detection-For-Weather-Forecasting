import argparse
import os
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

try:
    from src.config import (
        BASE_DIR,
        BATCH_SIZE,
        CLASSES,
        EPOCHS,
        LEARNING_RATE,
        MODEL_NAME,
        MODELS_DIR,
        NUM_CLASSES,
        NUM_WORKERS,
        PROCESSED_DATA_DIR,
        WEIGHT_DECAY,
    )
    from src.dataset import get_dataloaders
    from src.evaluate import evaluate_model
    from src.model import build_model, count_total_parameters, count_trainable_parameters
    from src.utils import (
        EarlyStopping,
        amp_enabled_for_device,
        calculate_classification_metrics,
        create_summary_writer,
        ensure_dir,
        get_device,
        history_to_rows,
        load_checkpoint,
        plot_confusion_matrix,
        plot_training_curves,
        save_checkpoint,
        save_csv,
        save_json,
        save_metrics_outputs,
        set_seed,
    )
except ImportError:
    from config import (
        BASE_DIR,
        BATCH_SIZE,
        CLASSES,
        EPOCHS,
        LEARNING_RATE,
        MODEL_NAME,
        MODELS_DIR,
        NUM_CLASSES,
        NUM_WORKERS,
        PROCESSED_DATA_DIR,
        WEIGHT_DECAY,
    )
    from dataset import get_dataloaders
    from evaluate import evaluate_model
    from model import build_model, count_total_parameters, count_trainable_parameters
    from utils import (
        EarlyStopping,
        amp_enabled_for_device,
        calculate_classification_metrics,
        create_summary_writer,
        ensure_dir,
        get_device,
        history_to_rows,
        load_checkpoint,
        plot_confusion_matrix,
        plot_training_curves,
        save_checkpoint,
        save_csv,
        save_json,
        save_metrics_outputs,
        set_seed,
    )


OUTPUT_DIR = Path(BASE_DIR) / "outputs" / "training"
BEST_MODEL_PATH = Path(MODELS_DIR) / "best_model.pth"
LAST_CHECKPOINT_PATH = Path(MODELS_DIR) / "last_checkpoint.pth"
TENSORBOARD_DIR = OUTPUT_DIR / "tensorboard"


def create_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    train_dir = os.path.join(PROCESSED_DATA_DIR, "train")
    val_dir = os.path.join(PROCESSED_DATA_DIR, "val")
    test_dir = os.path.join(PROCESSED_DATA_DIR, "test")
    return get_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def train_one_epoch(model, data_loader, criterion, optimizer, device, scaler=None, use_amp=False):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        autocast_context = torch.autocast(device_type="cuda") if use_amp else nullcontext()
        with autocast_context:
            outputs = model(images)
            loss = criterion(outputs, labels)

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        predictions = torch.argmax(outputs, dim=1)
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        correct += (predictions == labels).sum().item()
        total += batch_size

    average_loss = running_loss / total if total else 0.0
    accuracy = correct / total if total else 0.0
    return average_loss, accuracy


def run_training(args):
    set_seed(args.seed)
    ensure_dir(OUTPUT_DIR)
    ensure_dir(MODELS_DIR)

    device = get_device()
    use_amp = amp_enabled_for_device(device) and not args.disable_amp
    print(f"Device: {device}")
    print(f"AMP enabled: {use_amp}")

    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_model(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    early_stopping = EarlyStopping(patience=args.patience)
    writer = create_summary_writer(TENSORBOARD_DIR)

    start_epoch = 0
    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}

    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler, map_location=device)
        start_epoch = checkpoint.get("epoch", -1) + 1
        best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
        best_val_accuracy = checkpoint.get("best_val_accuracy", best_val_accuracy)
        history = checkpoint.get("metrics", {}).get("history", history)
        print(f"Resumed from checkpoint: {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler=scaler,
            use_amp=use_amp,
        )
        val_loss, val_accuracy, val_targets, val_predictions = evaluate_model(
            model,
            val_loader,
            criterion,
            device,
        )
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)

        if writer is not None:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/validation", val_loss, epoch)
            writer.add_scalar("Accuracy/train", train_accuracy, epoch)
            writer.add_scalar("Accuracy/validation", val_accuracy, epoch)
            writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)

        metrics = {
            "history": history,
            "validation": calculate_classification_metrics(val_targets, val_predictions, CLASSES),
        }

        save_checkpoint(
            LAST_CHECKPOINT_PATH,
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_loss,
            CLASSES,
            metrics=metrics,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_checkpoint(
                BEST_MODEL_PATH,
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_loss,
                CLASSES,
                metrics=metrics,
            )

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_accuracy:.4f} val_acc={val_accuracy:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.6g}"
        )

        if early_stopping.step(val_loss):
            print("Early stopping triggered.")
            break

    if writer is not None:
        writer.close()

    test_loss, test_accuracy, test_targets, test_predictions = evaluate_model(
        model,
        test_loader,
        criterion,
        device,
    )
    test_metrics = calculate_classification_metrics(test_targets, test_predictions, CLASSES)
    test_metrics["loss"] = test_loss
    test_metrics["accuracy"] = test_accuracy

    save_metrics_outputs(test_metrics, OUTPUT_DIR)
    save_json(OUTPUT_DIR / "training_history.json", history)
    save_csv(
        OUTPUT_DIR / "training_history.csv",
        history_to_rows(history),
        ["epoch", "train_loss", "val_loss", "train_accuracy", "val_accuracy"],
    )
    plot_training_curves(history, OUTPUT_DIR)
    plot_confusion_matrix(test_metrics["confusion_matrix"], CLASSES, OUTPUT_DIR / "confusion_matrix.png")

    return {
        "device": str(device),
        "amp_enabled": use_amp,
        "model_name": MODEL_NAME,
        "num_classes": NUM_CLASSES,
        "total_parameters": count_total_parameters(model),
        "trainable_parameters": count_trainable_parameters(model),
        "best_model_path": str(BEST_MODEL_PATH),
        "last_checkpoint_path": str(LAST_CHECKPOINT_PATH),
        "output_dir": str(OUTPUT_DIR),
    }


def run_verification(args):
    set_seed(args.seed)
    ensure_dir(OUTPUT_DIR)
    ensure_dir(MODELS_DIR)

    device = get_device()
    use_amp = amp_enabled_for_device(device) and not args.disable_amp
    print(f"Device: {device}")
    print(f"AMP enabled: {use_amp}")

    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Train images: {len(train_loader.dataset)}")
    print(f"Validation images: {len(val_loader.dataset)}")
    print(f"Test images: {len(test_loader.dataset)}")

    model = build_model(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
    )
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    images, labels = next(iter(train_loader))
    images = images.to(device)
    labels = labels.to(device)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    autocast_context = torch.autocast(device_type="cuda") if use_amp else nullcontext()
    with autocast_context:
        outputs = model(images)
        loss = criterion(outputs, labels)

    verification_checkpoint_path = OUTPUT_DIR / "phase3_verification_checkpoint.pth"
    if scaler is not None and use_amp:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    save_checkpoint(
        verification_checkpoint_path,
        model,
        optimizer,
        scheduler,
        epoch=0,
        best_val_loss=float(loss.item()),
        classes=CLASSES,
        metrics={"verification_loss": float(loss.item())},
    )

    verification = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "amp_enabled": use_amp,
        "model_name": MODEL_NAME,
        "num_classes": NUM_CLASSES,
        "classes": CLASSES,
        "train_images": len(train_loader.dataset),
        "val_images": len(val_loader.dataset),
        "test_images": len(test_loader.dataset),
        "batch_image_shape": list(images.shape),
        "batch_label_shape": list(labels.shape),
        "forward_output_shape": list(outputs.shape),
        "verification_loss": float(loss.item()),
        "checkpoint_saved": verification_checkpoint_path.exists(),
        "checkpoint_path": str(verification_checkpoint_path),
        "total_parameters": count_total_parameters(model),
        "trainable_parameters": count_trainable_parameters(model),
    }
    save_json(OUTPUT_DIR / "phase3_verification.json", verification)
    print("Phase 3 short verification completed.")
    print(verification)
    return verification


def parse_args():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 weather classifier.")
    parser.add_argument("--verify-only", action="store_true", help="Run one mini-batch verification only.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.verify_only:
        run_verification(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
