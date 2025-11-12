from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor

from src.data.dataset import VehicleAttributesDataset
from src.models.cnn import VehicleAttributeCNN
from src.models.transformer import VehicleAttributeTransformer
from src.utils.training import head_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained TwinCar model on labeled data.")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Training run directory containing checkpoints, config_used.yaml, and label_mappings.json",
    )
    parser.add_argument(
        "--model-type",
        choices=["cnn", "transformer"],
        required=True,
        help="Model architecture to load.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path. Defaults to <run_dir>/checkpoints/best_model.pt",
    )
    parser.add_argument(
        "--metadata-csv",
        type=str,
        default=None,
        help="CSV with image paths + labels. Defaults to the metadata path stored in the run config.",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default=None,
        help="Root folder containing images. Defaults to the image_root from the run config.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally evaluate on a random subset of this many samples.",
    )
    return parser.parse_args()


def load_config(run_dir: Path) -> Dict:
    config_path = run_dir / "config_used.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config_used.yaml under {run_dir}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_label_mappings(run_dir: Path) -> Dict[str, Dict[str, int]]:
    mapping_path = run_dir / "label_mappings.json"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Missing label_mappings.json under {run_dir}")
    with mapping_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def encode_labels(
    df: pd.DataFrame,
    label_columns: Dict[str, str],
    label_mappings: Dict[str, Dict[str, int]],
) -> pd.DataFrame:
    for head, column in label_columns.items():
        if column not in df.columns:
            raise ValueError(f"metadata CSV missing column '{column}' required for head '{head}'")
        if head not in label_mappings:
            raise ValueError(f"label_mappings.json missing head '{head}'")
        mapping = label_mappings[head]
        canonical = df[column].astype(str).fillna("Unknown")
        unseen = sorted(set(canonical.unique()) - set(mapping.keys()))
        if unseen:
            raise ValueError(
                f"Found labels not seen during training for head '{head}': {unseen[:5]}"
                + ("..." if len(unseen) > 5 else "")
            )
        df[f"{head}_id"] = canonical.map(mapping)
    return df


def build_cnn_transform(config: Dict) -> transforms.Compose:
    aug = config.get("augmentations", {})
    resize = aug.get("resize", 256)
    crop_size = aug.get("crop_size", 224)
    mean = aug.get("mean", [0.485, 0.456, 0.406])
    std = aug.get("std", [0.229, 0.224, 0.225])
    return transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_transformer_transform(run_dir: Path):
    processor_path = run_dir / "processor"
    if not processor_path.exists():
        raise FileNotFoundError(
            f"Cannot find saved processor at {processor_path}. "
            "Ensure you ran train_transformer.py for this run_dir."
        )
    processor = AutoImageProcessor.from_pretrained(processor_path)

    def _transform(image):
        encoded = processor(images=image, return_tensors="pt")
        return encoded["pixel_values"].squeeze(0)

    return _transform


def load_model(
    model_type: str,
    config: Dict,
    label_mappings: Dict[str, Dict[str, int]],
    device: torch.device,
) -> nn.Module:
    num_classes = {head: len(mapping) for head, mapping in label_mappings.items()}
    if model_type == "cnn":
        model = VehicleAttributeCNN(
            in_channels=config["model"].get("in_channels", 3),
            base_filters=config["model"]["base_filters"],
            num_blocks=config["model"]["num_blocks"],
            dropout=config["model"].get("dropout", 0.3),
            num_classes=num_classes,
        )
    else:
        checkpoint_name = config["model"]["checkpoint"]
        model = VehicleAttributeTransformer(
            model_name=checkpoint_name,
            num_classes=num_classes,
            dropout=config["model"].get("dropout", 0.1),
            freeze_backbone=False,
        )
    return model.to(device)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    heads,
) -> Dict[str, Dict[str, float]]:
    criterion = nn.CrossEntropyLoss()
    losses = []
    y_true = {head: [] for head in heads}
    y_pred = {head: [] for head in heads}
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = {head: target.to(device) for head, target in labels.items()}
            outputs = model(images)
            loss = sum(criterion(outputs[head], labels[head]) for head in heads)
            losses.append(loss.item())
            for head in heads:
                y_true[head].extend(labels[head].cpu().tolist())
                y_pred[head].extend(outputs[head].argmax(dim=1).cpu().tolist())
    head_stats = {head: head_metrics(y_true[head], y_pred[head]) for head in heads}
    composite = float(np.mean([metrics["f1"] for metrics in head_stats.values()]))
    return {"loss": float(np.mean(losses)), "per_head": head_stats, "macro_f1": composite}


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    config = load_config(run_dir)
    label_mappings = load_label_mappings(run_dir)

    metadata_path = Path(args.metadata_csv) if args.metadata_csv else Path(config["data"]["metadata_csv"])
    image_root = Path(args.image_root) if args.image_root else Path(config["data"]["image_root"])
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata CSV not found: {metadata_path}")
    if not image_root.exists():
        raise FileNotFoundError(f"image root not found: {image_root}")

    df = pd.read_csv(metadata_path)
    label_columns = config["data"]["label_columns"]
    df = encode_labels(df, label_columns, label_mappings)
    if args.limit and args.limit < len(df):
        df = df.sample(n=args.limit, random_state=42).reset_index(drop=True)

    if args.model_type == "cnn":
        transform = build_cnn_transform(config)
    else:
        transform = build_transformer_transform(run_dir)

    dataset = VehicleAttributesDataset(df, image_root, label_columns, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_type, config, label_mappings, device)
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else run_dir / "checkpoints" / "best_model.pt"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state"])

    results = evaluate(model, dataloader, device, list(label_columns.keys()))

    print(f"Evaluated {len(dataset)} samples from {metadata_path}")
    print(f"Average loss: {results['loss']:.4f}")
    print(f"Macro F1 (avg across heads): {results['macro_f1']:.4f}")
    for head, metrics in results["per_head"].items():
        print(
            f"[{head}] "
            f"acc={metrics['accuracy']:.4f} "
            f"prec={metrics['precision']:.4f} "
            f"rec={metrics['recall']:.4f} "
            f"f1={metrics['f1']:.4f}"
        )


if __name__ == "__main__":
    main()
