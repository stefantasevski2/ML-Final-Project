from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoImageProcessor

from src.models.cnn import VehicleAttributeCNN
from src.models.transformer import VehicleAttributeTransformer


class PredictionDataset(Dataset):
    def __init__(self, image_paths: List[str], transform) -> None:
        self.paths = [Path(p) for p in image_paths]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        if not path.exists():
            raise FileNotFoundError(f"Missing image: {path}")
        image = Image.open(path).convert("RGB")
        return self.transform(image), str(path)


def load_label_mappings(run_dir: Path) -> Dict[str, Dict[str, int]]:
    mapping_path = run_dir / "label_mappings.json"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Missing label mappings at {mapping_path}")
    with mapping_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def invert_mappings(mappings: Dict[str, Dict[str, int]]) -> Dict[str, Dict[int, str]]:
    return {
        head: {index: label for label, index in labels.items()}
        for head, labels in mappings.items()
    }


def build_cnn_transform(cfg: Dict) -> transforms.Compose:
    aug = cfg.get("augmentations", {})
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
    processor = AutoImageProcessor.from_pretrained(run_dir / "processor")

    def _transform(image):
        encoded = processor(images=image, return_tensors="pt")
        return encoded["pixel_values"].squeeze(0)

    return _transform, processor


def load_model(model_type: str, config: Dict, label_mappings: Dict[str, Dict[str, int]], device):
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
        model = VehicleAttributeTransformer(
            model_name=config["model"]["checkpoint"],
            num_classes=num_classes,
            dropout=config["model"].get("dropout", 0.1),
            freeze_backbone=False,
        )
    return model.to(device)


def parse_args():
    parser = argparse.ArgumentParser(description="Batch inference for trained models.")
    parser.add_argument("--run-dir", type=str, required=True, help="Training run directory.")
    parser.add_argument(
        "--model-type",
        choices=["cnn", "transformer"],
        required=True,
        help="Model family to load.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint. Defaults to run_dir/checkpoints/best_model.pt",
    )
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="List of image files to run inference on.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    config_path = run_dir / "config_used.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config_used.yaml in {run_dir}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    label_mappings = load_label_mappings(run_dir)
    inv_mappings = invert_mappings(label_mappings)

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
    model.eval()

    if args.model_type == "cnn":
        transform = build_cnn_transform(config)
    else:
        transform, _ = build_transformer_transform(run_dir)

    dataset = PredictionDataset(args.images, transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    results = []
    with torch.no_grad():
        for batch in dataloader:
            images, paths = batch
            images = images.to(device)
            outputs = model(images)
            predictions = {
                head: logits.argmax(dim=1).cpu().tolist() for head, logits in outputs.items()
            }
            for idx, image_path in enumerate(paths):
                prediction = {
                    head: inv_mappings[head][predictions[head][idx]]
                    for head in inv_mappings.keys()
                }
                results.append({"image": image_path, "prediction": prediction})

    for item in results:
        print(f"{item['image']}: {item['prediction']}")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
