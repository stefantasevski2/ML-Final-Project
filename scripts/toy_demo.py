from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import VehicleAttributesDataset, prepare_dataframe
from src.models.cnn import VehicleAttributeCNN

VEHICLE_CLASSES: List[Tuple[str, str, str]] = [
    ("Audi", "S4 Sedan", "2012"),
    ("BMW", "X5 SUV", "2007"),
    ("Chevrolet", "Camaro Convertible", "2012"),
    ("Tesla", "Model S Sedan", "2012"),
    ("Toyota", "Corolla Sedan", "2012"),
    ("Mercedes-Benz", "C-Class Sedan", "2012"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a toy multi-label dataset and run a quick inference demo."
    )
    parser.add_argument("--root", type=str, default="data/toy", help="Output directory for toy data.")
    parser.add_argument("--num-samples", type=int, default=12, help="How many synthetic images to create.")
    parser.add_argument("--image-size", type=int, default=256, help="Image height/width for generated samples.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of mini epochs for the demo CNN fit.")
    parser.add_argument("--batch-size", type=int, default=4, help="Mini-batch size used in the training loop.")
    return parser.parse_args()


def create_toy_dataset(root: Path, num_samples: int, image_size: int) -> Path:
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = root / "metadata.csv"

    palette = [
        (200, 50, 50),
        (250, 170, 50),
        (60, 150, 220),
        (50, 200, 120),
        (140, 80, 200),
        (240, 240, 70),
    ]
    rng = np.random.default_rng(1234)
    rows = []

    for idx in range(num_samples):
        make, model, year = VEHICLE_CLASSES[idx % len(VEHICLE_CLASSES)]
        color = palette[idx % len(palette)]
        filename = f"toy_car_{idx:03d}.png"
        image_path = images_dir / filename
        synthesize_image(image_path, make, model, color, image_size, rng)
        rows.append(
            {
                "image_path": filename,
                "make": make,
                "model": model,
                "year": year,
            }
        )

    with metadata_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["image_path", "make", "model", "year"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created {num_samples} toy samples under {root}")
    return metadata_path


def synthesize_image(
    path: Path,
    make: str,
    model: str,
    base_color: Tuple[int, int, int],
    size: int,
    rng: np.random.Generator,
) -> None:
    array = np.ones((size, size, 3), dtype=np.uint8) * np.array(base_color, dtype=np.uint8)
    noise = rng.integers(0, 40, size=(size, size, 3), dtype=np.uint8)
    array = np.clip(array + noise, 0, 255).astype(np.uint8)
    image = Image.fromarray(array)

    draw = ImageDraw.Draw(image)
    annotation = f"{make}\n{model}"
    try:
        font = ImageFont.truetype("arial.ttf", size // 12)
    except OSError:
        font = ImageFont.load_default()
    spacing = 4
    try:
        bbox = draw.multiline_textbbox((0, 0), annotation, font=font, spacing=spacing)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except AttributeError:
        lines = annotation.split("\n")
        line_sizes = [draw.textsize(line, font=font) for line in lines]
        text_w = max(width for width, _ in line_sizes)
        text_h = sum(height for _, height in line_sizes) + spacing * (len(lines) - 1)
    draw.rectangle(
        [(5, size - text_h - 10), (text_w + 15, size - 5)],
        fill=(0, 0, 0, 180),
    )
    draw.multiline_text(
        (10, size - text_h - 8),
        annotation,
        font=font,
        spacing=spacing,
        fill=(255, 255, 255),
    )
    image.save(path)


def run_demo_training(
    metadata_csv: Path,
    image_root: Path,
    epochs: int,
    batch_size: int,
) -> None:
    label_columns = {"make": "make", "model": "model", "year": "year"}
    df, label_mappings = prepare_dataframe(metadata_csv, label_columns)

    dataset = VehicleAttributesDataset(
        dataframe=df,
        image_root=image_root,
        label_names=label_columns,
        transform=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        ),
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VehicleAttributeCNN(
        in_channels=3,
        base_filters=16,
        num_blocks=3,
        dropout=0.2,
        num_classes={head: len(mapping) for head, mapping in label_mappings.items()},
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    heads = list(label_mappings.keys())
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for images, labels in dataloader:
            images = images.to(device)
            labels = {head: target.to(device) for head, target in labels.items()}
            optimizer.zero_grad()
            outputs = model(images)
            loss = sum(criterion(outputs[head], labels[head]) for head in heads)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"[epoch {epoch + 1}/{epochs}] loss={epoch_loss / len(dataloader):.4f}")

    demonstrate_inference(model, dataloader, label_mappings, device)


def demonstrate_inference(model, dataloader, label_mappings, device):
    model.eval()
    inv_maps = {
        head: {idx: label for label, idx in head_map.items()}
        for head, head_map in label_mappings.items()
    }
    with torch.no_grad():
        images, _ = next(iter(dataloader))
        images = images.to(device)
        outputs = model(images)
        predictions = {head: logits.argmax(dim=1).cpu().tolist() for head, logits in outputs.items()}
    for idx in range(len(images)):
        decoded = {
            head: inv_maps[head][predictions[head][idx]]
            for head in inv_maps.keys()
        }
        print(f"Sample {idx}: {decoded}")


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    metadata_csv = create_toy_dataset(root, args.num_samples, args.image_size)
    run_demo_training(
        metadata_csv=metadata_csv,
        image_root=root / "images",
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
