from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor

from src.data.dataset import (
    VehicleAttributesDataset,
    prepare_dataframe,
    split_dataframe,
)
from src.models.transformer import VehicleAttributeTransformer
from src.utils.config import load_yaml_config, save_json
from src.utils.training import head_metrics, save_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a Hugging Face vision backbone for vehicle attributes."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/transformer.yaml",
        help="Path to YAML config.",
    )
    return parser.parse_args()


def build_transform(processor):
    def _transform(image):
        encoded = processor(images=image, return_tensors="pt")
        return encoded["pixel_values"].squeeze(0)

    return _transform


def create_dataloaders(cfg, transform):
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
        transform=transform,
    )
    val_dataset = VehicleAttributesDataset(
        val_df,
        cfg["data"]["image_root"],
        {head: cfg["data"]["label_columns"][head] for head in label_heads},
        transform=transform,
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
    metrics = {head: head_metrics(y_true[head], y_pred[head]) for head in heads}
    return {"loss": float(np.mean(losses)), "per_head": metrics}


def main():
    args = parse_args()
    config = load_yaml_config(args.config)
    set_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoImageProcessor.from_pretrained(config["model"]["checkpoint"])
    transform = build_transform(processor)
    train_loader, val_loader, label_mappings = create_dataloaders(config, transform)

    num_classes = {head: len(mapping) for head, mapping in label_mappings.items()}
    model = VehicleAttributeTransformer(
        model_name=config["model"]["checkpoint"],
        num_classes=num_classes,
        dropout=config["model"].get("dropout", 0.1),
        freeze_backbone=config["model"].get("freeze_backbone", False),
    ).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 0.0),
        betas=(0.9, 0.999),
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
    processor.save_pretrained(run_dir / "processor")
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
        composite_score = np.mean([m["f1"] for m in val_metrics["per_head"].values()])
        print(
            f"Epoch {epoch}: train_loss={epoch_log['train_loss']:.4f} "
            f"val_f1={composite_score:.4f}"
        )

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


if __name__ == "__main__":
    main()
