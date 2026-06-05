# Reproducao de segmentacao em colonoscopia

Reproducao do artigo "Application of Deep Learning Models for Semantic Segmentation of Colonoscopy Images".

## Uso

```bash
make setup
make train MODEL=cascade
```

Modelos em `upstream/commands.yaml`.

ESFPNet local:

```bash
uv run python -m scripts.train --config configs/models/esfpnet.yaml
```
