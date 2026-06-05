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

## Gerar a Tabela 2

A tabela e calculada a partir das mascaras preditas no split de teste. O formato esperado para qualquer modelo e:

```text
outputs/predictions/<modelo>/<stem_da_imagem>.png
```

Depois do treino, exporte as predicoes do modelo desejado:

```bash
make predict MODEL=esfpnet
make predict MODEL=cascade
```

Cada exportador salva as mascaras em `outputs/predictions/<modelo>/`. Depois gere a tabela:

```bash
make table2
```

Os arquivos sao salvos em `outputs/tables/table2.csv` e `outputs/tables/table2.md`.
