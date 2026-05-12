# colonoscopy-segmentation-reproduction
Reproduction of the paper: Application of Deep Learning Models for Semantic Segmentation of Colonoscopy Images

## Data

Download and prepare Kvasir-SEG with:

```bash
uv run python -m scripts.download_dataset --dataset kvasir
```

The command saves the downloaded archive in `data/downloads/` and prepares the dataset at:

```text
data/raw/kvasir-seg/
  images/
  masks/
```

Then validate the data loading with:

```bash
uv run python -m scripts.validate_data --data_root data/raw/kvasir-seg
```
