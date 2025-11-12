from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


@dataclass
class DatasetConfig:
    metadata_csv: Path
    image_root: Path
    label_columns: Dict[str, str]
    val_split: float
    random_state: int = 42
    stratify_column: Optional[str] = None


class VehicleAttributesDataset(Dataset):
    """Dataset that returns an image tensor and label dictionary."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_root: Path,
        label_names: Dict[str, str],
        transform=None,
    ) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.label_names = label_names
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        record = self.df.iloc[index]
        image_path = (self.image_root / record["image_path"]).expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        labels = {
            head: torch.tensor(int(record[f"{head}_id"]), dtype=torch.long)
            for head in self.label_names.keys()
        }
        return image, labels


def prepare_dataframe(
    metadata_csv: Path, label_columns: Dict[str, str]
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    """
    Loads a metadata CSV and appends *_id columns for each supervised head.
    """
    metadata_path = Path(metadata_csv)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")

    df = pd.read_csv(metadata_path)
    required_columns = {"image_path", *label_columns.values()}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Metadata CSV missing columns: {sorted(missing)}")

    label_mappings: Dict[str, Dict[str, int]] = {}
    for head, column in label_columns.items():
        series = df[column].fillna("Unknown").astype(str)
        categories = sorted(series.unique())
        mapping = {category: idx for idx, category in enumerate(categories)}
        df[f"{head}_id"] = series.map(mapping)
        label_mappings[head] = mapping

    return df, label_mappings


def split_dataframe(
    df: pd.DataFrame, val_split: float, random_state: int, stratify_column: Optional[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < val_split < 1:
        raise ValueError("val_split must be between 0 and 1 (exclusive).")

    stratify = df[stratify_column] if stratify_column else None
    train_df, val_df = train_test_split(
        df, test_size=val_split, random_state=random_state, stratify=stratify
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
