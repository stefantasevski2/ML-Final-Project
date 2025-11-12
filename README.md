# TwinCar Visios

Minimal reference for multi-label vehicle recognition (make, model, year) using either a scratch CNN or a pretrained transformer.

## Layout

```
.
├── configs/             # YAML configs for each training recipe
├── data/                # Place images + metadata.csv here
├── outputs/             # Training/eval artefacts (created at runtime)
├── scripts/             # Toy demo helpers
├── src/                 # Shared data/models/utils modules
├── train_cnn.py         # Scratch CNN trainer
├── train_transformer.py # HF transformer fine-tuner
├── predict.py           # Batch inference
└── evaluate.py          # Metric reporting on labeled CSVs
```

## General instructions

```bash
pip install -r requirements.txt

# train scratch CNN
python train_cnn.py --config configs/cnn.yaml

# fine-tune HF backbone
python train_transformer.py --config configs/transformer.yaml

# run evaluation on a labeled split
python evaluate.py --model-type cnn --run-dir outputs/cnn/<run> --metadata-csv data/val.csv --image-root data/images

# batch inference (any model)
python predict.py --model-type cnn --run-dir outputs/cnn/<run> --images sample1.jpg sample2.jpg --output-json preds.json
```

See `data/README.md` for the expected CSV schema (image path + make/model/year columns).

## Minimal Python Example

```python
import torch
from src.models.cnn import VehicleAttributeCNN

num_classes = {"make": 10, "model": 50, "year": 5}
model = VehicleAttributeCNN(in_channels=3, base_filters=32, num_blocks=4, num_classes=num_classes)
dummy = torch.randn(2, 3, 224, 224)
outputs = model(dummy)
print({head: logits.shape for head, logits in outputs.items()})
```
