# colonoscopy-segmentation-reproduction
Reproduction of the paper: Application of Deep Learning Models for Semantic Segmentation of Colonoscopy Images

## Data

Download and prepare Kvasir-SEG with:

```bash
uv run python scripts/download_dataset.py
```

The command downloads the archive into a temporary directory and prepares the dataset at:

```text
data/raw/kvasir-seg/
  images/
  masks/
```

Then validate the data loading with:

```bash
uv run python scripts/validate_data.py --data_root data/raw/kvasir-seg
```

## Models

ESFPNet is available through the model registry:

```python
from models import get_model

model = get_model("esfpnet", model_type="b0", num_classes=1)
```

Supported ESFPNet variants are `b0`, `b1`, `b2`, `b3`, `b4`, and `b5`.
Pass `pretrained_path="path/to/mit_b0.pth"` if you have local pretrained MiT
encoder weights.
