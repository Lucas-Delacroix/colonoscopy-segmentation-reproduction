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
