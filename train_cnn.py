from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.dataset import (
    VehicleAttributesDataset,
    prepare_dataframe,
    split_dataframe,
)
from src.models.cnn import VehicleAttributeCNN
from src.utils.config import load_yaml_config, save_json
from src.utils.training import head_metrics, save_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train custom CNN for vehicle attributes.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cnn.yaml",
        help="Path to YAML config.",
    )
    return parser.parse_args()


def build_transforms(cfg: Dict) -> Dict[str, transforms.Compose]:
    aug = cfg.get("augmentations", {})
    resize = aug.get("resize", 256)
    crop_size = aug.get("crop_size", 224)
    mean = aug.get("mean", [0.485, 0.456, 0.406])
    std = aug.get("std", [0.229, 0.224, 0.225])

    train_tfms = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(p=aug.get("horizontal_flip", 0.5)),
            transforms.ColorJitter(
                brightness=aug.get("brightness", 0.2),
                contrast=aug.get("contrast", 0.2),
                saturation=aug.get("saturation", 0.2),
                hue=aug.get("hue", 0.05),
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    val_tfms = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return {"train": train_tfms, "val": val_tfms}


def create_dataloaders(
    cfg: Dict, transforms_map: Dict[str, transforms.Compose]
) -> tuple[DataLoader, DataLoader, Dict[str, Dict[str, int]]]:
    df, label_mappings = prepare_dataframe(
        cfg["data"]["metadata_csv"], cfg["data"]["label_columns"]
    )
    train_df, val_df = split_dataframe(
        df,
        val_split=cfg["data"]["val_split"],
        random_state=cfg.get("seed", 42),
        stratify_column=cfg["data"].get("stratify_column"),
    )
    label_heads = list(cfg["data"]["label_columns"].keys())
    train_dataset = VehicleAttributesDataset(
        train_df,
        cfg["data"]["image_root"],
        {head: cfg["data"]["label_columns"][head] for head in label_heads},
        transform=transforms_map["train"],
    )
    val_dataset = VehicleAttributesDataset(
        val_df,
        cfg["data"]["image_root"],
        {head: cfg["data"]["label_columns"][head] for head in label_heads},
        transform=transforms_map["val"],
    )
    training_cfg = cfg["training"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=True,
        num_workers=training_cfg.get("num_workers", 4),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=False,
        num_workers=training_cfg.get("num_workers", 4),
        pin_memory=True,
    )
    return train_loader, val_loader, label_mappings


def aggregate_metrics(
    y_true: Dict[str, List[int]], y_pred: Dict[str, List[int]]
) -> Dict[str, Dict[str, float]]:
    return {head: head_metrics(y_true[head], y_pred[head]) for head in y_true.keys()}


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    set_seed(config.get("seed", 42))

    transforms_map = build_transforms(config)
    train_loader, val_loader, label_mappings = create_dataloaders(config, transforms_map)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = {head: len(mapping) for head, mapping in label_mappings.items()}
    model = VehicleAttributeCNN(
        in_channels=config["model"].get("in_channels", 3),
        base_filters=config["model"]["base_filters"],
        num_blocks=config["model"]["num_blocks"],
        dropout=config["model"].get("dropout", 0.3),
        num_classes=num_classes,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 0.0),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["epochs"]
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=config["training"].get("mixed_precision", False))

    run_dir = Path(config["logging"]["output_dir"]) / datetime.utcnow().strftime(
        "%Y%m%d-%H%M%S"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    with (run_dir / "config_used.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)
    save_json(label_mappings, run_dir / "label_mappings.json")

    heads = list(label_mappings.keys())
    best_score = -float("inf")
    history = []

    for epoch in range(1, config["training"]["epochs"] + 1):
        model.train()
        train_losses = []
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = {head: label.to(device) for head, label in targets.items()}
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=config["training"].get("mixed_precision", False)):
                outputs = model(images)
                loss = sum(criterion(outputs[head], targets[head]) for head in heads)
            scaler.scale(loss).backward()
            if config["training"].get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["training"]["grad_clip"]
                )
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())
        scheduler.step()

        val_metrics = evaluate(model, val_loader, device, heads, criterion)
        epoch_log = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_log)
        print(f"Epoch {epoch}: train_loss={epoch_log['train_loss']:.4f} val_f1="
              f"{np.mean([v['f1'] for v in val_metrics['per_head'].values()]):.4f}")

        composite_score = np.mean([m["f1"] for m in val_metrics["per_head"].values()])
        if composite_score > best_score:
            best_score = composite_score
            save_checkpoint(
                {"model_state": model.state_dict(), "epoch": epoch},
                run_dir / "checkpoints" / "best_model.pt",
            )

        if epoch % config["logging"].get("save_every", 1) == 0:
            save_checkpoint(
                {"model_state": model.state_dict(), "epoch": epoch},
                run_dir / "checkpoints" / f"epoch_{epoch}.pt",
            )

    save_json({"history": history}, run_dir / "training_history.json")
    print(f"Training completed. Outputs stored in {run_dir}")


def evaluate(model, dataloader, device, heads, criterion):
    model.eval()
    losses = []
    y_true = {head: [] for head in heads}
    y_pred = {head: [] for head in heads}
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            targets = {head: label.to(device) for head, label in targets.items()}
            outputs = model(images)
            loss = sum(criterion(outputs[head], targets[head]) for head in heads)
            losses.append(loss.item())
            for head in heads:
                y_true[head].extend(targets[head].cpu().tolist())
                y_pred[head].extend(outputs[head].argmax(dim=1).cpu().tolist())
    metrics = aggregate_metrics(y_true, y_pred)
    return {"loss": float(np.mean(losses)), "per_head": metrics}


if __name__ == "__main__":
    main()
